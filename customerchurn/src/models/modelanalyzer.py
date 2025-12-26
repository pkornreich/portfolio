# BS"D

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Use ImbPipeline for resampling steps
from sklearn.model_selection import StratifiedKFold, cross_val_score

from models.modelobjects import MetaModel
from models.modelobjects import ModelData
from models.modelobjects import ModelResult


from rich import print

class ModelAnalyzer:

    @staticmethod
    def get_scores(model_meta: MetaModel, data: ModelData, use_smote: bool=False) -> ModelResult:
        X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2, random_state=42, stratify=data.y)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        model = model_meta.model
        pipeline = ImbPipeline([
            ('classifier', model)
        ])

        if use_smote:
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', model)
            ])
        
        # 3. Run the Cross-Validation on the pipeline
        precision_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring='precision',  # Or 'accuracy', 'recall', 'f1'
            n_jobs=-1
        )

        precision_cv = precision_scores.mean()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        result = ModelResult(model_meta.name, data.data_description, precision_cv, accuracy,
            precision, recall, f1, cm, use_smote)
        
        return result

