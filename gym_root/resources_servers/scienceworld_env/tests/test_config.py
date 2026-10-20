"""configs/scienceworld.yaml must validate under Gym's server-config schema *offline*.

Gym refuses to start an "almost-server" (a block that has an entrypoint but fails
pydantic validation) and the whole gym boot dies with it; every server block is
validated here the same way global_config.py does it (alfworld_env precedent).
"""

from copy import deepcopy
from pathlib import Path

from nemo_gym.config_types import is_almost_server, maybe_get_server_instance_config
from omegaconf import OmegaConf

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "scienceworld.yaml"


def _server_blocks(cfg):
    return {name: block for name, block in cfg.items() if is_almost_server(block)}


def test_every_server_block_validates_under_gym_schema():
    blocks = _server_blocks(OmegaConf.load(CONFIG))
    assert set(blocks) == {"scienceworld_env", "scienceworld_agent"}
    for name, block in blocks.items():
        _config, error = maybe_get_server_instance_config(name=name, server_type_config_dict=block)
        assert error is None, f"{name} in {CONFIG.name} fails Gym validation:\n{error}"


def test_validation_catches_a_bad_domain_value():
    cfg = deepcopy(OmegaConf.load(CONFIG))
    name = "scienceworld_env"
    cfg[name]["resources_servers"][name]["domain"] = "web"
    _config, error = maybe_get_server_instance_config(name=name, server_type_config_dict=cfg[name])
    assert error is not None and "domain" in str(error)


def test_yaml_matches_server_config_fields():
    """Every knob in the yaml is a real config field (a typo would be silently
    ignored by pydantic extra handling or rejected at boot, depending on Gym)."""
    from scienceworld_env.app import ScienceworldResourcesServerConfig

    block = OmegaConf.to_container(OmegaConf.load(CONFIG))["scienceworld_env"]["resources_servers"]["scienceworld_env"]
    known = set(ScienceworldResourcesServerConfig.model_fields)
    unknown = {k for k in block if k not in known and k not in {"entrypoint", "domain", "verified", "description"}}
    assert not unknown, unknown
    assert block["session_ttl_s"] > 1200, "must exceed NEMO_GYM_RUN_TIMEOUT"
