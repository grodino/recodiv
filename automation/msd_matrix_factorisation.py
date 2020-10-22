from automan.api import Problem


class CreateQuadriPartiteGraph(Problem):
    """Create the tripartite graph from """
    def get_name(self):
        return 'create_quadri_partite_graph'

    def get_commands(self):
        return [
            ('1', f'python recodiv/msd_matrix_factorisation.py --output-dir "{self.input_path()}"', None),
        ]

    def run(self):
        self.make_output_dir()