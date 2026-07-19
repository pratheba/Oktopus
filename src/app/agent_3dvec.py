# Compatibility shim. The monolithic `Agent` was split into:
#   agent_3dvec_base (AgentBase: field-agnostic infra)
#   agent_3dvec_sdf  (AgentSDF: signed-distance; MC/RFTA/carve)
#   agent_3dvec_udf  (AgentUDF: unsigned-distance; DualMeshUDF only)
from agent_3dvec_sdf import AgentSDF
from agent_3dvec_udf import AgentUDF
from agent_3dvec_sdf_as_udf import AgentSDFasUDF
Agent = AgentSDF
