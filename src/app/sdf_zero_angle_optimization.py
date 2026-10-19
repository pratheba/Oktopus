"""0045: provable no-op auto snug elimination for zero-angular-follow wraps.

Applies only to the legacy multiplicative snug mode: in the mapping,
    scale = scale_iso * angular_ratio ** wrap_angular_follow
and the snug field affects only angular_ratio. At zero exponent, the
snug field cannot affect the sample coordinates. Additive snug, field
handoff and attach-to-free blending deliberately stay on the old path.
"""


def can_skip_inert_snug(adapt_arg):
    if not adapt_arg.get('auto_avatar_snug_field', False):
        return False
    if adapt_arg.get('mode', 'direct') != 'direct':
        return False
    if not adapt_arg.get('wrap_radius', False) or adapt_arg.get('rigid_radius', False):
        return False
    if float(adapt_arg.get('wrap_angular_follow', 1.0)) != 0.0:
        return False
    if str(adapt_arg.get('snug_mode', 'multiplicative')).lower() != 'multiplicative':
        return False
    if adapt_arg.get('avatar_snug_scale_field') is not None:
        return False
    if adapt_arg.get('shared_snug_field'):
        return False
    if adapt_arg.get('use_attach_to_free_blend',
                     adapt_arg.get('use_wrap_to_rigid_blend', False)):
        return False
    if adapt_arg.get('snug_penetration_guard', False):
        return False
    if adapt_arg.get('wrap_debug', False):
        return False
    return True
