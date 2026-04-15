    def build_ring_line_geometry(
        self,
        curve: Array,
        radius: Array,
        every: int = 5,
        n_ring: int = 48,
        close_ring: bool = True,
    ) -> Tuple[Array, List[List[int]]]:
        """
        Returns OBJ-style line geometry for radius rings.
        """
        curve = np.asarray(curve, dtype=np.float64)
        radius = np.asarray(radius, dtype=np.float64)

        T, N, B = self.compute_rmf(curve)

        verts = []
        lines = []

        for i in range(0, len(curve), max(1, every)):
            ring = self._circle3d(curve[i], N[i], B[i], float(radius[i]), n=n_ring)
            start = len(verts)
            verts.extend(ring.tolist())
            ids = list(range(start, start + len(ring)))
            if close_ring:
                ids = ids + [ids[0]]
            lines.append(ids)

        if len(verts) == 0:
            return np.zeros((0, 3), dtype=np.float64), []

        return np.asarray(verts, dtype=np.float64), lines
