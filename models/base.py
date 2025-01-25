# monitor/models/base.py
class BaseClassifier:
    def check(self, text):
        raise NotImplementedError
