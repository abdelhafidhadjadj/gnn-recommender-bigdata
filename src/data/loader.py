"""Raw CSV loading — accepts either config-based paths or explicit file paths.

Two loading backends are provided:
  load_raw_data()   — pandas (standard mode, world_size=1)
  load_via_spark()  — PySpark (bigdata mode, world_size > 1)

The correct loader is selected automatically by prepare_data() in main.py
based on the DDP world_size.  Both return the same tuple
(business_df, user_df, review_df) so the rest of the pipeline is unchanged.
"""
import os
import pandas as pd
from config import DataConfig


# ── pandas loader (standard, w=1) ─────────────────────────────────────────────

def load_raw_data(cfg: DataConfig,
                  business_path: str = None,
                  user_path: str = None,
                  review_path: str = None):
    """
    Load the three Yelp CSVs via pandas.
    Explicit *_path arguments override cfg.data_dir + filename.
    """
    def resolve(explicit, fname):
        return explicit if explicit else os.path.join(cfg.data_dir, fname)

    business_df = pd.read_csv(resolve(business_path, cfg.business_file))
    user_df     = pd.read_csv(resolve(user_path,     cfg.user_file))
    review_df   = pd.read_csv(resolve(review_path,   cfg.review_file))
    if cfg.max_users   > 0: user_df   = user_df.head(cfg.max_users)
    if cfg.max_reviews > 0: review_df = review_df.head(cfg.max_reviews)
    return business_df, user_df, review_df


# ── Spark loader (bigdata, w > 1) ─────────────────────────────────────────────

def load_via_spark(
    cfg: DataConfig,
    rank: int = 0,
) -> tuple:
    """
    Load the three Yelp CSVs via PySpark (bigdata distributed mode).

    Each DDP rank launches its own local SparkSession so the load is
    fully parallel across workers — mirroring how a real multi-node Spark
    cluster would distribute the read.  Spark web UI is disabled to avoid
    port conflicts when N processes start simultaneously on the same host.

    If the environment variable SPARK_MASTER is set (e.g. to
    "spark://spark-master:7077" from docker-compose.bigdata.yml), the
    external cluster is used instead of local mode.

    Returns the same (business_df, user_df, review_df) tuple as
    load_raw_data() so the rest of the pipeline is unchanged.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError(
            "[Spark] pyspark is not installed.\n"
            "  pip install pyspark\n"
            "  or switch back to pandas mode (world_size=1)."
        )

    spark_master = os.environ.get("SPARK_MASTER", "local[1]")
    hdfs_url     = os.environ.get("HDFS_URL", "")      # e.g. hdfs://namenode:9000

    builder = (
        SparkSession.builder
        .appName(f"GNN-Rec-Load-rank{rank}")
        .master(spark_master)
        # Disable web UI — N processes on the same host would fight for port 4040
        .config("spark.ui.enabled", "false")
        # Small shuffle partition count (we just read, not shuffle)
        .config("spark.sql.shuffle.partitions", "4")
        # Driver heap: 4g supports full dataset (188k reviews, 145k users)
        # multiple JVMs coexist — each DDP rank gets its own SparkSession
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        # Désactiver Arrow pour toPandas() → retourne des numpy arrays standard
        # (Arrow retourne des ArrowStringArray qui cassent torch.from_numpy)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    )

    if hdfs_url:
        # Configure Hadoop FS so Spark can talk to the HDFS namenode
        builder = (
            builder
            .config("spark.hadoop.fs.defaultFS", hdfs_url)
            .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # ── Resolve file paths (HDFS CSV ou local CSV) ───────────────────────────
    if hdfs_url:
        # Données uploadées via scripts/hdfs_upload.sh
        # Layout HDFS : hdfs://namenode:9000/data/{size}/yelp_*.csv
        size_tag = os.path.basename(os.path.normpath(cfg.data_dir))  # e.g. "1k"
        base     = f"{hdfs_url}/data/{size_tag}"
        storage  = f"HDFS CSV ({base})"
    else:
        base    = cfg.data_dir
        storage = "local filesystem (CSV)"

    def resolve(fname: str) -> str:
        return f"{base}/{fname}" if hdfs_url else os.path.join(base, fname)

    # ── Read & materialise (trigger the actual I/O) ───────────────────────────
    biz_sdf = spark.read.option("header", "true").csv(resolve(cfg.business_file))
    usr_sdf = spark.read.option("header", "true").csv(resolve(cfg.user_file))
    rev_sdf = spark.read.option("header", "true").csv(resolve(cfg.review_file))

    # Cache + count forces Spark to actually read the files right now
    # (lazy evaluation would otherwise defer it to toPandas()).
    biz_sdf.cache(); usr_sdf.cache(); rev_sdf.cache()
    n_biz = biz_sdf.count()
    n_usr = usr_sdf.count()
    n_rev = rev_sdf.count()

    if rank == 0:
        print(
            f"[Spark] rank={rank}  master={spark_master}  storage={storage}\n"
            f"        businesses={n_biz:,}  users={n_usr:,}  reviews={n_rev:,}"
        )

    # ── Apply size limits (same behaviour as pandas .head()) ─────────────────
    if cfg.max_users   > 0: usr_sdf = usr_sdf.limit(cfg.max_users)
    if cfg.max_reviews > 0: rev_sdf = rev_sdf.limit(cfg.max_reviews)

    # ── Convert to pandas for the rest of the (PyTorch) pipeline ─────────────
    business_df = biz_sdf.toPandas()
    user_df     = usr_sdf.toPandas()
    review_df   = rev_sdf.toPandas()

    spark.stop()

    # ── Forcer les types numériques ───────────────────────────────────────────
    # Spark lit les CSV en string ; pd.read_csv infère les types automatiquement.
    # On aligne les dtypes pour que la suite du pipeline (graph_builder, etc.)
    # reçoive les mêmes types qu'avec load_raw_data().
    for col in ["stars", "review_count"]:
        if col in business_df.columns:
            business_df[col] = pd.to_numeric(business_df[col], errors="coerce")
    for col in ["review_count", "average_stars"]:
        if col in user_df.columns:
            user_df[col] = pd.to_numeric(user_df[col], errors="coerce")
    for col in ["stars"]:
        if col in review_df.columns:
            review_df[col] = pd.to_numeric(review_df[col], errors="coerce")

    return business_df, user_df, review_df
