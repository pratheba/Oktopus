    def build_cylinder_tube_mesh(
        self,
        curve: Array,
        radius: Array,
        n_theta: int = 24,
        cap_ends: bool = False,
    ) -> Tuple[Array, Array]:
        """
        Build a simple triangle tube around the curve using RMF.
        """
        curve = np.asarray(curve, dtype=np.float64)
        radius = np.asarray(radius, dtype=np.float64)

        if len(curve) < 2:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

        T, N, B = self.compute_rmf(curve)
        th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

        verts = np.zeros((len(curve) * n_theta, 3), dtype=np.float64)

        for i in range(len(curve)):
            ring = curve[i][None] + radius[i] * (
                np.cos(th)[:, None] * N[i][None] +
                np.sin(th)[:, None] * B[i][None]
            )
            verts[i * n_theta:(i + 1) * n_theta] = ring

        faces = []
        for i in range(len(curve) - 1):
            base0 = i * n_theta
            base1 = (i + 1) * n_theta
            for j in range(n_theta):
                jn = (j + 1) % n_theta
                a = base0 + j
                b = base0 + jn
                c = base1 + j
                d = base1 + jn
                faces.append([a, c, b])
                faces.append([b, c, d])

        if cap_ends:
            start_center = len(verts)
            end_center = len(verts) + 1
            verts = np.vstack([verts, curve[0], curve[-1]])

            for j in range(n_theta):
                jn = (j + 1) % n_theta
                faces.append([start_center, jn, j])
                a = (len(curve) - 1) * n_theta + j
                b = (len(curve) - 1) * n_theta + jn
                faces.append([end_center, a, b])

        return verts, np.asarray(faces, dtype=np.int32)
