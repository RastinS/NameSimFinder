from django.apps import AppConfig


class FinderAgentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'finderAgent'

    def ready(self):
        from .services import SimFinderModel
        finderModel = SimFinderModel()
        finderModel.load()
