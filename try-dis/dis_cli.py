import dis
import csv
import sys

def disassemble_file(input_file):
    """
    Disassembles a Python file and extracts information about each instruction.

    Args:
        input_file (str): The path to the input Python file.

    Returns:
        list: A list of dictionaries containing information about each disassembled instruction.
    """
    instructions = []
    with open(input_file, 'r') as f:
        code = compile(f.read(), input_file, 'exec')
        for instr in dis.get_instructions(code):
            instructions.append({
                'offset': instr.offset,
                'opname': instr.opname,
                'opcode': instr.opcode,
                'arg': instr.arg,
                'argval': instr.argval,
                'argrepr': instr.argrepr,
                'lineno': instr.starts_line
            })
    return instructions

def write_to_csv(instructions, output_file):
    """
    Writes the disassembled instructions to a CSV file.

    Args:
        instructions (list): A list of dictionaries containing information about each disassembled instruction.
        output_file (str): The path to the output CSV file.
    """
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['offset', 'opname', 'opcode', 'arg', 'argval', 'argrepr', 'lineno']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for instr in instructions:
            writer.writerow(instr)

def main():
    """
    Main function of the disassembler script.
    Parses command line arguments, disassembles the input Python file, and writes the instructions to a CSV file.
    """
    if len(sys.argv) != 3:
        print("Usage: python disassembler.py <input_file.py> <output_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    instructions = disassemble_file(input_file)
    write_to_csv(instructions, output_file)
    print(f"Disassembled instructions written to {output_file}")

if __name__ == "__main__":
    main()

