"""0047 shared opt-in selection for validated SDF streaming configurations.

The evaluator is common to rigid and wrap mapping.  Composition remains an
explicit semantic choice: an independent piece or a validated two-section
post-SDF cone seam.  Fail closed for root/leaf and MC composition until they
can preserve those operations' existing semantics.
"""


def configure_unified_sdf_stream(cfg, items, *, extractor):
    """Return a new config with the supported shared streaming path selected.

    This does not change adaptation YAML entries or coordinate mapping.
    """
    cfg = dict(cfg)
    if not bool(cfg.get('sdf_stream_unified', False)):
        return cfg
    if extractor != 'ours':
        raise ValueError('0047 shared streaming currently supports DCSDD ours only')
    if cfg.get('root_leaves') is not None:
        raise ValueError('0047 root_leaves attachment/ownership composition is not yet implemented; use legacy root_leaves pipeline')
    if cfg.get('dcsdd_stream_accepted_count') is not None:
        raise ValueError('0047 shared streaming discovers accepted samples automatically; remove accepted-count hint')
    if not cfg.get('dcsdd_stream_replay_dir') and not cfg.get('sdf_grid_memmap_dir'):
        # Runtime MCGrid can carry a memmap_dir, so don't reject if only that
        # is present; caller enforces scratch availability when opening stage.
        pass
    combined = bool(cfg.get('combine_adaptations', False))
    if (len(items) == 1 and not combined
            and items[0].get('mode', 'direct') == 'direct'
            and items[0].get('rigid_radius', False)
            and not items[0].get('wrap_radius', False)):
        if not bool(cfg.get('export_individual_meshes', True)):
            raise ValueError('0047 independent streaming requires export_individual_meshes: true')
        if not items[0].get('use_accessory_support_clamp', False):
            raise ValueError('0047 independent streaming requires accessory support clamp')
        if cfg.get('sdf_stream_combined_wrap', False):
            raise ValueError('0047 independent rigid adaptation cannot use combined-wrap streaming')
        for key in ('dcsdd_bounded_filtering', 'dcsdd_fast_bbox_marking',
                    'dcsdd_stream_inference', 'dcsdd_stream_crop_first',
                    'dcsdd_stream_replay_metadata', 'dcsdd_stream_one_pass',
                    'dcsdd_crop_grid', 'sdf_localize_after_gate'):
            cfg[key] = True
        mode = 'independent_rigid'
    elif (combined and len(items) == 2 and all(
            item.get('mode', 'direct') == 'direct'
            and item.get('wrap_radius', False)
            and not item.get('rigid_radius', False) for item in items)):
        if bool(cfg.get('export_individual_meshes', True)):
            raise ValueError('0047 combined wrap streaming requires export_individual_meshes: false')
        if any(cfg.get(k, False) for k in (
                'dcsdd_stream_inference', 'dcsdd_stream_one_pass',
                'dcsdd_stream_crop_first', 'dcsdd_stream_replay_metadata')):
            raise ValueError('0047 combined wrap uses the shared writer, not the independent rigid boot flags')
        for key in ('sdf_stream_combined_wrap', 'sdf_localize_after_gate',
                    'sdf_optimize_zero_angular_wrap', 'dcsdd_crop_grid',
                    'dcsdd_fast_bbox_marking'):
            cfg[key] = True
        mode = 'combined_wrap_two_piece'
    else:
        raise ValueError(
            '0047 shared streaming supports independent rigid and validated '
            'two-section combined wrap only; mixed wrap/rigid, MC and '
            'root_leaves still require the legacy composition path')
    print('[sdf_stream_unified] mode=', mode,
          'mapping=', ('rigid' if mode == 'independent_rigid' else 'wrap'),
          'composition=', ('independent' if not combined else 'post_sdf_seam'),
          flush=True)
    return cfg
