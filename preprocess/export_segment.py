    def export_segment_debug_geometry(
        self,
        seg: SegmentView,
        out_dir: str,
        every_ring: int = 5,
        n_ring: int = 48,
        tube_theta: int = 24,
    ) -> List[str]:
        """
        Export 3D geometry for one segment:
          - raw curve
          - center curve
          - wrap curve
          - key curve
          - train radius rings
          - wrap radius rings
          - cylinder radius rings
          - cylinder tube
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        sid = seg.segment_id

        # 1) curves as line OBJ
        curve_sets = {
            "raw_curve": seg.polyline_raw,
            "center_curve": seg.polyline_center,
            "wrap_curve": seg.polyline_wrap,
            "key_curve": seg.keypoints,
        }

        for name, curve in curve_sets.items():
            if len(curve) == 0:
                continue
            p = out_dir / f"part_{sid:03d}_{name}.obj"
            self._write_obj_lines(
                p,
                np.asarray(curve, dtype=np.float64),
                [list(range(len(curve)))]
            )
            paths.append(str(p))

        # 2) rings on the SAME key curve
        ring_specs = {
            "train_rings": seg.key_radius_train,
            "wrap_rings": seg.key_radius_wrap,
            "cylinder_rings": seg.cylinder_radius,
        }

        for name, radius in ring_specs.items():
            verts, lines = self.build_ring_line_geometry(
                curve=seg.keypoints,
                radius=radius,
                every=every_ring,
                n_ring=n_ring,
            )
            if len(verts):
                p = out_dir / f"part_{sid:03d}_{name}.obj"
                self._write_obj_lines(p, verts, lines)
                paths.append(str(p))

        # 3) cylinder tube mesh
        tube_v, tube_f = self.build_cylinder_tube_mesh(
            curve=seg.keypoints,
            radius=seg.cylinder_radius,
            n_theta=tube_theta,
            cap_ends=False,
        )
        if len(tube_v):
            p = out_dir / f"part_{sid:03d}_cylinder_tube.obj"
            self._write_obj_mesh(p, tube_v, tube_f)
            paths.append(str(p))

        return paths


    def export_all_segments_debug_geometry(
        self,
        segments: List[SegmentView],
        out_dir: str,
        every_ring: int = 5,
        n_ring: int = 48,
        tube_theta: int = 24,
    ) -> List[str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for seg in segments:
            paths.extend(
                self.export_segment_debug_geometry(
                    seg=seg,
                    out_dir=str(out_dir / "parts"),
                    every_ring=every_ring,
                    n_ring=n_ring,
                    tube_theta=tube_theta,
                )
            )
        return paths
