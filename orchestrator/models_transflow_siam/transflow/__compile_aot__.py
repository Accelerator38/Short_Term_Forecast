
if __name__ == '__main__':
    import sys
    from pathlib import Path
    project_root = Path.cwd().parent.parent
    sys.path.insert(0, str(project_root.resolve()))

    from models_transflow_siam.transflow import analytical_model
    from models_transflow_siam.transflow import analytical_optimize
    analytical_model.compile_file()
    analytical_optimize.compile_file()
