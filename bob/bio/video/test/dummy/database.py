from bob.bio.base.database import ZTBioDatabase
from bob.bio.base.test.utils import atnt_database_directory


class DummyDatabase(ZTBioDatabase):

    def __init__(self):
        # call base class constructor with useful parameters
        super(DummyDatabase, self).__init__(
            name='test',
            original_directory=atnt_database_directory(),
            original_extension='.pgm',
            check_original_files_for_existence=True,
            training_depends_on_protocol=False,
            models_depend_on_protocol=False
        )
        import bob.db.atnt
        self.__db = bob.db.atnt.Database()

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self.__db.model_ids(groups, protocol)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        return self.__db.objects(model_ids, groups, purposes, protocol, **kwargs)

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        return []

    def zobjects(self, groups=None, protocol=None, **kwargs):
        return []

    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        return self.__db.model_ids(groups)

    def t_enroll_files(self, t_model_id, group='dev'):
        return self.enroll_files(t_model_id, group)

    def z_probe_files(self, group='dev'):
        return self.probe_files(None, group)

    # override all_files to return a one-element lists of files
    def all_files(self, groups):
        return [[n] for n in super(DummyDatabase, self).all_files(groups)]

    def file_names(self, files, directory, extension):
        if isinstance(files[0], list):
            files = list(list(zip(*files))[0])
        return super(DummyDatabase, self).file_names(files, directory, extension)

database = DummyDatabase()
