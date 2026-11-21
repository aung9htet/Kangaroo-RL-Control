"""README.md contains no fictional commands: every command parses with the real CLI, for all ten controllers."""
import re
import shlex
from pathlib import Path

import pytest

from natural_rl.cli import NOT_YET, build_parser
from natural_rl.models.registry import MODEL_IDS

RUNBOOK = (Path(__file__).resolve().parents[2] / 'README.md').read_text()
EXPECTED_ORDER = ['train', 'checkpoints', 'play', 'video', 'video', 'video', 'eval', 'analyse', 'plot', 'export', 'results']


def sections():
    parts = re.split(r'^### ', RUNBOOK, flags=re.M)[1:]
    return {p.split(':')[0].strip(): p for p in parts}


def commands(text):
    block = re.search(r'```bash\n(.*?)```', text, flags=re.S).group(1)
    return [line for line in block.splitlines() if line.startswith('python -m natural_rl')]


def parse(command):
    tokens = shlex.split(command)[3:]
    args, extra = build_parser().parse_known_args(tokens)
    if args.command not in NOT_YET:
        assert not extra, f'{command!r} has arguments the CLI does not accept: {extra}'
    return args


def test_every_registered_controller_has_its_own_heading():
    heads = sections()
    for model in MODEL_IDS:
        assert model in heads, f'RUNBOOK is missing the heading for {model}'


@pytest.mark.parametrize('model', MODEL_IDS)
def test_each_controller_block_has_the_eleven_commands_and_all_of_them_parse(model):
    text = sections()[model]
    cmds = commands(text)
    assert len(cmds) == 11, f'{model}: {len(cmds)} commands'
    parsed = [parse(c) for c in cmds]
    assert [p.command for p in parsed] == EXPECTED_ORDER
    train = parsed[0]
    assert train.model == model and train.run_id == f'{model}-r7-s00' and train.checkpoint_every == 50000
    assert train.record_every_checkpoint and train.steps == 1_000_000 and train.reward == 'r7_balanced'
    assert [p.action for p in parsed[3:6]] == ['record', 'backfill', 'verify']
    assert parsed[3].suite == 'checkpoint_video_v1' and parsed[3].checkpoint_step == 150000
    assert parsed[5].all_checkpoints and parsed[2].checkpoint_step == 150000 and parsed[2].task_id == 'val_001'
    assert all(getattr(p, 'run_id', train.run_id) == train.run_id for p in parsed)
    assert parsed[6].manifest == 'configs/eval/test_v1.json'
    if model == 'sac_pidlag':
        assert train.cost_budget_id == 'budget_moderate_v1'
    else:
        assert train.cost_budget_id is None


def test_each_block_names_its_paper_and_the_output_layout_is_documented_once():
    for model in MODEL_IDS:
        text = sections()[model]
        assert re.search(r'https?://', text) and f'configs/models/{model}.yaml' in text
    assert 'results/runs/<backend>/<model>/<reward>/<run_id>/' in RUNBOOK


def test_the_preamble_commands_parse_too():
    head = RUNBOOK.split('## 3. The ten controllers')[0]
    commands_found = re.findall(r'^python -m natural_rl .*$', head, flags=re.M)
    assert len(commands_found) >= 5
    for command in commands_found:
        tokens = shlex.split(re.sub(r'\s+#.*$', '', command))
        if '|' in tokens:                                       # a pipe outside quotes: only the natural_rl command before it is ours
            tokens = tokens[:tokens.index('|')]
        parse(shlex.join(tokens))
