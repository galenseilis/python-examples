import argparse
import importlib.util
from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput

def main():
    parser = argparse.ArgumentParser(description='Profile Python code using pycallgraph2.')
    parser.add_argument('source_file', help='Python source file containing the code to profile')
    parser.add_argument('output_file', help='Output file name (without extension) to save the profiling graph')
    args = parser.parse_args()

    source_file = args.source_file
    output_filename = args.output_file

    # Load the module from the provided source file
    spec = importlib.util.spec_from_file_location("module", source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Run pycallgraph on the entire module
    with PyCallGraph(output=GraphvizOutput(output_file=f"{output_filename}.png")):
        # Execute the entire module
        exec(open(source_file).read(), module.__dict__)

if __name__ == "__main__":
    main()

