import joblib

def read_model(fit_model_path):

    if type(fit_model_path) == str:
        if fit_model_path[-4:] == ".pkl":
            with open(fit_model_path, 'rb') as f:
                model = pickle.load(f)
        elif fit_model_path[-7:] == ".joblib":
            model = joblib.load(fit_model_path)
        return model

    elif type(fit_model_path) == dict:
        models = {}
        for model_path, weight in fit_model_path.items():
            if model_path[-4:] == ".pkl":
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                models[model] = weight
            elif model_path[-7:] == ".joblib":
                model = joblib.load(fit_model_path)
                models[model] = weight
        return models