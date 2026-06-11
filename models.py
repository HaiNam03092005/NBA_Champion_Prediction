import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_voting_baseline(X_train, y_top4, y_top2, y_champ):
    """Mô hình 1: Voting Ensemble Cơ bản"""
    base_learners = [
        ('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
    ]
    
    model_top4 = VotingClassifier(estimators=base_learners, voting='soft')
    model_top4.fit(X_train, y_top4)
    
    model_top2 = VotingClassifier(estimators=base_learners, voting='soft')
    model_top2.fit(X_train, y_top2)
    
    model_champ = VotingClassifier(estimators=base_learners, voting='soft')
    model_champ.fit(X_train, y_champ)
    
    return model_top4, model_top2, model_champ

def train_stacking_baseline(X_train, y_top4, y_top2, y_champ):
    """Mô hình 2: Stacking Tiêu chuẩn (Dùng Logistic Regression)"""
    base_learners = [
        ('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
    ]
    
    meta_learner = LogisticRegression(class_weight='balanced', random_state=42)
    
    model_top4 = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
    model_top4.fit(X_train, y_top4)
    
    model_top2 = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
    model_top2.fit(X_train, y_top2)
    
    model_champ = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
    model_champ.fit(X_train, y_champ)
    
    return model_top4, model_top2, model_champ