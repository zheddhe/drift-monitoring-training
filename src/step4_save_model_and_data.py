import os
# import pandas as pd
from joblib import dump


def save_model_and_data(model, df):

    # Create 'model' directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # save the model
    with open('./models/lin_reg.bin', 'wb') as f_out:
        dump(model, f_out)

    # save referenced data (validation data)
    df.to_parquet('data/reference.parquet')


if __name__ == "__main__":
    pass
