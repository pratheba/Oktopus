import numpy as np
from agent_3dvec_udf import AgentUDF

class AgentSDFasUDF(AgentUDF):
    """Diagnostic ONLY: an OLD *signed* checkpoint reused as a fake UDF via
    abs(SDF) -> DualMeshUDF. Never use for a real UDF-trained model."""

    def _udf_clamp(self, x):
        return np.abs(np.asarray(x, dtype=np.float64))
