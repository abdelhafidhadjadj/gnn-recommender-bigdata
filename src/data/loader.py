"""Raw CSV loading — accepts either config-based paths or explicit file paths."""
import os
import pandas as pd
from config import DataConfig


def load_raw_data(cfg: DataConfig,
                  business_path: str = None,
                  user_path: str = None,
                  review_path: str = None):
    """
    Load the three Yelp CSVs.  Explicit *_path arguments override cfg.data_dir + filename.
    """
    def resolve(explicit, fname):
        return explicit if explicit else os.path.join(cfg.data_dir, fname)

    business_df = pd.read_csv(resolve(business_path, cfg.business_file))
    user_df     = pd.read_csv(resolve(user_path,     cfg.user_file)).head(cfg.max_users)
    review_df   = pd.read_csv(resolve(review_path,   cfg.review_file)).head(cfg.max_reviews)
    return business_df, user_df, review_df
