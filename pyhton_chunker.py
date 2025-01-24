import ast
import os

def get_python_chunks(file_path):
    """
    Erstellt Chunks basierend auf:
    1) Jede top-level Function -> eigener Chunk (type="function")
    2) Jede Methode innerhalb einer Klasse -> eigener Chunk (type="class_method")
    3) Alles übrige top-level Code (z.B. Imports, globale Variablen etc.) wird
       in einem einzigen Chunk zusammengefasst (type="module"), falls vorhanden.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    lines = source.splitlines()
    tree = ast.parse(source, mode='exec')

    chunks = []

    # Wir sammeln "restlichen" Code (nicht ClassDef/FunctionDef) als Module-Chunk
    # => Dafür merken wir uns die Start-/End-Zeilen aller "anderen" Nodes und fassen sie
    #    am Ende in EINEN Chunk zusammen.
    leftover_start = None
    leftover_end = None

    def flush_leftover():
        """Erzeugt einen einzelnen Modul-Chunk (falls noch Code übrig ist)."""
        nonlocal leftover_start, leftover_end
        if leftover_start is not None and leftover_end is not None:
            chunk_source = "\n".join(lines[leftover_start:leftover_end])
            chunks.append({
                "type": "module",
                "name": "module_chunk",
                "source": chunk_source,
                "file_path": file_path,
                "start_line": leftover_start,
                "end_line": leftover_end
            })
            leftover_start = None
            leftover_end = None

    # Wir gehen nur die TOP-LEVEL-Nodes durch (tree.body),
    # damit wir wirklich nur Klassen, top-level Funktionen und sonstige Statements bekommen.
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Zuerst flushen wir evtl. vorhandenen Modul-Code,
            # der VOR dieser Klasse steht.
            flush_leftover()

            # Für jede Methode (FunctionDef) im Körper der Klasse -> eigener Chunk
            for class_body_node in node.body:
                if isinstance(class_body_node, ast.FunctionDef):
                    start_line = class_body_node.lineno - 1
                    end_line = class_body_node.end_lineno
                    function_source = "\n".join(lines[start_line:end_line])
                    chunks.append({
                        "type": "class_method",
                        "name": f"{node.name}.{class_body_node.name}",
                        "source": function_source,
                        "file_path": file_path,
                        "start_line": start_line,
                        "end_line": end_line
                    })

        elif isinstance(node, ast.FunctionDef):
            # Wir flushen den zuvor gesammelten Modul-Code bis hier.
            flush_leftover()

            # Dies ist eine Top-Level-Funktion
            start_line = node.lineno - 1
            end_line = node.end_lineno
            function_source = "\n".join(lines[start_line:end_line])
            chunks.append({
                "type": "function",
                "name": node.name,
                "source": function_source,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line
            })

        else:
            # Alle anderen Node-Arten sind Teil des „restlichen Module-Codes“
            # Wir mergen sie in einen Bereich (von leftover_start bis leftover_end).
            this_start = node.lineno - 1
            this_end = node.end_lineno
            if leftover_start is None:
                leftover_start = this_start
            # Wir setzen leftover_end immer wieder auf das Maximum 
            # (sodass wir einen zusammenhängenden Bereich abdecken).
            leftover_end = max(leftover_end or 0, this_end)

    # Ganz am Ende flushen wir evtl. noch vorhandenen Modul-Code.
    flush_leftover()

    return chunks

if __name__ == "__main__":
    folder_path = r"C:\Users\rudi\source\repos\Tools"

    # Get all python files in the folder and all subfolders.
    python_files = []
    for root, _, files in os.walk(folder_path):
        # do not include venv folders and .git folders or hidden folders
        if '.' in root or "venv" in root or ".git" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    all_chunks = []
    for python_file in python_files:
        file_path = os.path.join(folder_path, python_file)
        chunks = get_python_chunks(file_path)
        all_chunks.extend(chunks)

    for chunk in all_chunks:
        print(chunk["file_path"])
        print(f"{chunk['name']} : {chunk['type']} : {chunk['start_line']} : {chunk['end_line']}")
        print(chunk["source"])
        print("----\n\n")
