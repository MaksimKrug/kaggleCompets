from src.predict import predict


def test_predict_dtype(synthetic_data):
    """Check inference dtype"""
    preds = predict(synthetic_data)
    assert all((isinstance(p, float) for p in preds))


def test_predict_stability(synthetic_data):
    """Check inference stability"""
    pred1 = predict(synthetic_data)
    pred2 = predict(synthetic_data)
    assert all(x == y for x, y in zip(pred1, pred2))
