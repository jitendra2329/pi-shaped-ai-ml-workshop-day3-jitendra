from sklearn.feature_selection import SelectKBest, f_classif

def select_top_k_features(X, y, k=15):
    """
    Select top k features using ANOVA F-value classification test.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = X.columns[selected_indices].tolist()

    print(f"Top {k} selected features: {selected_feature_names}")
    return X_new, selected_feature_names, selector
