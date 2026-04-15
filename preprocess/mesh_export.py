    @staticmethod
    def _write_obj_lines(path: Path, vertices: Array, lines: List[List[int]]) -> None:
        """
        Write OBJ with vertices and polyline 'l' elements.
        `lines` contains 0-based vertex index lists.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for line in lines:
                idx = [str(i + 1) for i in line]  # OBJ is 1-based
                f.write("l " + " ".join(idx) + "\n")

    @staticmethod
    def _write_obj_mesh(path: Path, vertices: Array, faces: Array) -> None:
        """
        Write OBJ with triangular faces.
        faces are 0-based.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for tri in faces:
                a, b, c = tri + 1
                f.write(f"f {a} {b} {c}\n")
