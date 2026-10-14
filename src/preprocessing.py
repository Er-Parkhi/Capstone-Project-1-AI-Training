
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
 
 
def build_preprocessor():

    numeric_pipeline = Pipeline(

        steps=[

            ("imputer", SimpleImputer(strategy="median")),

            ("scaler", StandardScaler()),

        ]

    )
 
    categorical_pipeline = Pipeline(

        steps=[

            (

                "imputer",

                SimpleImputer(strategy="most_frequent"),

            ),

            (

                "onehot",

                OneHotEncoder(handle_unknown="ignore"),

            ),

        ]

    )
 
    return ColumnTransformer(

        transformers=[

            ("numeric", numeric_pipeline, NUMERIC_FEATURES),

            (

                "categorical",

                categorical_pipeline,

                CATEGORICAL_FEATURES,

            ),

        ],

        remainder="drop",

    )

