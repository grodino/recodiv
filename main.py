from automan.api import Automator

from automation.msd_matrix_factorisation import CreateQuadriPartiteGraph

def main():
    automator = Automator(
        simulation_dir='outputs',
        output_dir='figures',
        all_problems=[CreateQuadriPartiteGraph]
    )
    automator.run()

if __name__ == '__main__':
    main()