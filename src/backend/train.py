import os
import sys
import time
import tempfile

# Set JAVA_HOME if not already set and the path exists (local dev only).
# On other machines, set the JAVA_HOME environment variable manually.
java_path = os.environ.get("JAVA_HOME_OVERRIDE", r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot")
if not os.environ.get("JAVA_HOME") and os.path.exists(java_path):
    os.environ["JAVA_HOME"] = java_path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import BucketedRandomProjectionLSH

def main():
    print("Initialize Spark Session...")
    spark = SparkSession.builder \
        .appName("SpotifyRecommenderTraining") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "20") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.local.dir", os.path.join(tempfile.gettempdir(), "spark_tmp")) \
        .getOrCreate()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "spotify_millsongdata.csv")
    out_path = os.path.join(base_dir, "models", "similarity_matrix.parquet")
    features_out_path = os.path.join(base_dir, "models", "song_features.parquet")

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    import pandas as pd
    print(f"Loading data from {data_path} (Using fast-pandas parser)...")
    
    # PySpark's native multiline CSV parser is notoriously slow and causes stage hangs. 
    # Pandas parses 70MB in 0.5 seconds!
    pdf = pd.read_csv(data_path)
    pdf = pdf.dropna(subset=["artist", "song", "text"]).reset_index(drop=True)
    
    # We will sample 10,000 songs to ensure the matrix generates in < 2 minutes locally. 
    # (Feel free to increase this if your machine has the RAM!)
    pdf = pdf.sample(n=min(10000, len(pdf)), random_state=42)

    # Pre-clean text in pandas before sending to Spark.
    # Truncating lyrics to 2000 chars keeps task size well under 1000 KiB
    # while retaining enough signal for TF-IDF similarity.
    pdf["text"] = pdf["text"].astype(str).str.strip().str[:2000]
    pdf = pdf[["artist", "song", "text"]].reset_index(drop=True)
    
    print("Converting to PySpark DataFrame...")
    df = spark.createDataFrame(pdf)
    
    from pyspark.sql.functions import monotonically_increasing_id
    df = df.withColumn("song_id", monotonically_increasing_id())

    print("Building NLP Pipeline (TF-IDF)...")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=5000)
    idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures")
    
    normalizer = Normalizer(inputCol="idfFeatures", outputCol="features")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, normalizer])
    
    print("Fitting transform pipeline...")
    start_time = time.time()
    model = pipeline.fit(df)
    features_df = model.transform(df)

    features_df = features_df.select("song_id", "artist", "song", "features").cache()
    
    print(f"Pipeline finished! Materializing features for {features_df.count()} songs...")
    print(f"Time taken so far: {time.time() - start_time:.2f} seconds. Fitting LSH...")

    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0, numHashTables=3)
    model_lsh = brp.fit(features_df)

    print("Computing Approximate Similarity Join (This may take several minutes)...")
    start_join = time.time()
    
    similar_pairs = model_lsh.approxSimilarityJoin(features_df, features_df, 1.1, distCol="EuclideanDistance")

    filtered_pairs = similar_pairs.filter(col("datasetA.song_id") != col("datasetB.song_id"))

    print("Selecting the top 10 most similar songs for each track...")
    window = Window.partitionBy(col("datasetA.song_id")).orderBy(col("EuclideanDistance"))
    
    ranked_pairs = filtered_pairs.withColumn("rank", row_number().over(window))
    top_10_pairs = ranked_pairs.filter(col("rank") <= 10)

    final_df = top_10_pairs.select(
        col("datasetA.song").alias("song"),
        col("datasetA.artist").alias("artist"),
        col("datasetB.song").alias("similar_song"),
        col("datasetB.artist").alias("similar_artist"),
        col("EuclideanDistance").alias("distance"),
        col("rank")
    )

    print(f"Join query plan created in {time.time() - start_join:.2f} seconds. Saving matrix...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    final_pd = final_df.toPandas()
    final_pd.to_parquet(out_path, index=False)
    
    songs_ref = features_df.select("artist", "song").toPandas()
    songs_ref.to_parquet(features_out_path, index=False)

    print(f"Success! Pre-calculated similarity matrix saved to {out_path}")
    print(f"Song reference list saved to {features_out_path}")
    spark.stop()

    # --- sklearn: full-dataset feature vectors for live similarity fallback ---
    # Runs over ALL songs (not just the 10k Spark sample) so every song in the
    # dropdown has a vector for on-the-fly cosine similarity.
    print("\nBuilding full-dataset TF-IDF + SVD features for live similarity fallback...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    import numpy as np

    full_pdf = pd.read_csv(data_path)
    full_pdf = full_pdf.dropna(subset=["artist", "song", "text"]).reset_index(drop=True)
    full_pdf["text"] = full_pdf["text"].astype(str).str.strip().str[:2000]
    print(f"  Loaded {len(full_pdf)} songs for feature extraction...")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, sublinear_tf=True)
    X = vectorizer.fit_transform(full_pdf["text"])

    # 100-dim SVD (LSA) gives a dense, compact representation.
    # 100 dims × 57k songs ≈ 22MB parquet — lightweight enough to load in the UI.
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_svd = svd.fit_transform(X)
    X_norm = normalize(X_svd)   # L2-normalise: cosine similarity = dot product

    feat_df = full_pdf[["artist", "song"]].copy()
    feat_cols = pd.DataFrame(X_norm.astype("float32"), columns=[f"f{i}" for i in range(100)])
    feat_df = pd.concat([feat_df.reset_index(drop=True), feat_cols], axis=1)

    vectors_path = os.path.join(base_dir, "models", "song_vectors.parquet")
    feat_df.to_parquet(vectors_path, index=False)
    print(f"  Song vectors saved to {vectors_path}  ({len(feat_df)} songs, 100 dims)")

if __name__ == "__main__":
    main()
