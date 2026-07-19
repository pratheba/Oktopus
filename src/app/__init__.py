import sys
import os.path as op
current_path = op.dirname(op.abspath(__file__))
sys.path.append(current_path)
from agent_3dvec_sdf import AgentSDF
from agent_3dvec_udf import AgentUDF
from agent_3dvec_sdf_as_udf import AgentSDFasUDF
Agent = AgentSDF
