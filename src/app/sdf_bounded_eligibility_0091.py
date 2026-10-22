"""Conservative eligibility for 0035 DCSDD bounded localization.

This is a *performance* switch. Disabling it must not change an adaptation's
geometry, ownership, or SDF arithmetic. The separate cone callback stream
sets its own validated guard; ordinary action_part_adapt does not.
"""

GLOBAL_STATS_FLAGS = (
    'wrap_to_rigid', 'wrap_blend_rigid', 'unified_wrap',
    'use_free_profile_scale', 'use_attach_to_free_blend',
    'use_wrap_to_rigid_blend', 'free_anchor_to_release_scale',
)


def permitted_bounded_localization(adapt_arg, *, staged_wrap=False):
    """Return (is_permitted, reason) for an action_part_adapt adaptation."""
    if adapt_arg.get('mode', 'direct') != 'direct':
        return False, 'only direct adaptations have a verified bounded localizer'
    problematic = [key for key in GLOBAL_STATS_FLAGS if adapt_arg.get(key, False)]
    if problematic:
        return False, 'global-statistics blend: ' + ', '.join(problematic)
    rigid = bool(adapt_arg.get('rigid_radius', False))
    wrap = bool(adapt_arg.get('wrap_radius', False))
    if rigid and not wrap:
        return True, 'rigid direct'
    if wrap and not rigid and staged_wrap:
        return True, 'verified sparse root/leaf wrap staging'
    if wrap and not rigid:
        return False, 'ordinary wrap uses full localization; bounded wrap requires the separate validated stream or sparse staging'
    return False, 'no single supported radius mode'
