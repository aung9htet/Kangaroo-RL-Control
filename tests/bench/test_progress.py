"""Progress bar text, plain-mode output, message ordering, and the ordered run lines printed by `catalog programme --with-eval`."""
import io

from natural_rl.cli import build_parser, main
from natural_rl.programme import phase_runs, run_lines
from natural_rl.progress import Progress, fmt_time


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(tty, clock=None, **kw):
    stream = io.StringIO()
    clock = clock or FakeClock()
    return Progress('sac', 'sac-r7-s00', 'r7_balanced', 0, 1_000_000, 21, stream=stream, tty=tty, clock=clock, **kw), stream, clock


def test_line_names_the_model_run_reward_seed_and_shows_progress_speed_and_eta():
    p, _, clock = make(False)
    clock.t += 100.0
    p.saved = 8
    line = p.line(250_000)
    for part in ('sac | sac-r7-s00 | r7_balanced seed 0', ' 25.0% 250,000/1,000,000', 'ckpt 8/21', '2,500 tr/s', '0:01:40 elapsed', 'ETA 0:05:00'):
        assert part in line, part
    assert line.count('█') == 6 and line.count('░') == 18                       # 25 % of a 24-character bar


def test_phase_is_shown_while_a_video_is_recorded():
    p, _, _ = make(False)
    p.phase('recording video of step 150,000')
    assert p.line(10).endswith('recording video of step 150,000')


def test_plain_mode_prints_a_line_every_five_percent_not_every_step():
    p, stream, clock = make(False)
    for step in range(0, 1_000_001, 16_000):
        clock.t += 0.01
        p.update(step)
    lines = [l for l in stream.getvalue().splitlines() if l.startswith('[progress]')]
    assert 14 <= len(lines) <= 22 and len(lines) < 63 and '\r' not in stream.getvalue()   # ~1 per 5-7 %, not per step


def test_tty_mode_redraws_one_line_and_messages_appear_above_it():
    p, stream, clock = make(True)
    p.update(100_000)
    p.log('[checkpoint] step 100000 scheduled')
    out = stream.getvalue()
    assert out.count('\r') >= 2 and '\033[K' in out
    assert out.index('[checkpoint] step 100000 scheduled\n') < out.rindex('sac | sac-r7-s00')       # message first, bar redrawn below it


def test_finish_reports_status_time_and_checkpoint_count():
    p, stream, clock = make(False)
    clock.t += 3725
    p.step, p.saved = 1_000_000, 21
    p.finish('complete')
    assert 'complete after 1:02:05' in stream.getvalue() and '21/21 checkpoints' in stream.getvalue()


def test_fmt_time_handles_unknown():
    assert fmt_time(float('inf')) == '--:--:--' and fmt_time(59) == '0:00:59'


def test_run_lines_are_ordered_and_depend_on_the_phase():
    p0, p1, p2 = phase_runs('P0')[0], phase_runs('P1')[0], phase_runs('P2')[0]
    assert [l.split()[3] for l in run_lines(p0)] == ['train', 'video']
    assert [l.split()[3:5] for l in run_lines(p1)] == [['train'], ['video'], ['eval']] or len(run_lines(p1)) == 3
    assert '--final' in run_lines(p1)[2]
    kinds = [' '.join(l.split()[3:5]) for l in run_lines(p2)]
    assert kinds == ['train --model', 'video verify', 'eval --run-id', 'eval --run-id', 'analyse --run-id']
    assert '--all-checkpoints' in run_lines(p2)[2] and 'test_v1' in run_lines(p2)[3]
    assert run_lines(p2, with_eval=False) == run_lines(p2)[:1]


def test_every_printed_line_of_the_whole_programme_parses_with_the_real_cli(capsys):
    for phase in ('P0', 'P1', 'P2', 'P4'):
        assert main(['catalog', 'programme', '--phase', phase, '--with-eval']) == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith('python -m natural_rl')]
    assert len(lines) == 3 * 2 + 1440 * 3 + 2400 * 5 + 10 * 5                  # every cell of the action x state grid
    parser = build_parser()
    for line in lines:
        args, extra = parser.parse_known_args(line.split()[3:])
        assert not extra, line


def test_speed_and_eta_exclude_video_time_and_budget_the_videos_still_to_come():
    p, _, clock = make(False)
    p.phase('recording video of step 0')
    clock.t += 50.0                                        # a 50 s video before any training
    p.phase('')
    clock.t += 100.0                                       # then 100 s of training
    p.saved = 1
    line = p.line(100_000)                                 # 1,000 tr/s on training time (not 100,000/150)
    assert '1,000 tr/s' in line
    # remaining: 900,000 / 1,000 = 900 s of training + 20 videos x 50 s = 1,000 s  -> 0:31:40
    assert 'ETA 0:31:40' in line and '0:02:30 elapsed' in line
