from datetime import datetime
import functools
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
import multiprocessing as mp
from pathlib import Path
import sys
from tempfile import TemporaryFile
from urllib.parse import parse_qsl

import matplotlib.pyplot as plt
from yattag import Doc

from force.util import read_tfevents, try_parse
from force.workflow.filter import filter_check


def iter_subdirs(dir):
    assert dir.is_dir()
    for d in dir.iterdir():
        if d.is_dir():
            yield d


def relative_path(root_dir, path):
    root_dir_str = str(root_dir)
    path_str = str(path)
    assert path_str.startswith(root_dir_str)
    return path_str[len(root_dir_str):]


def parse_run(run: str):
    date_str, time_str, run_id = run.split('_')
    date_time_str = '_'.join([date_str, time_str])
    date_time = datetime.strptime(date_time_str, '%y-%m-%d_%H.%M.%S')
    return date_time, run_id


def index_page(directory,
               plot=None,
               status=None,
               domain=None, algorithm=None,
               id=None, days=None,
               **kwargs):
    # Preprocess args
    should_plot = plot is not None
    if should_plot: plot_keys = plot.split(',')
    incl_status = set(status.split(',')) if status is not None else None
    incl_domain = set(domain.split(',')) if domain is not None else None
    incl_algorithm = set(algorithm.split(',')) if algorithm is not None else None
    incl_ids = set(id.split(',')) if id is not None else None
    if days is not None: days = int(days)
    hp_filters = {k: set(try_parse(x) for x in v.split(',')) for k, v in kwargs.items()}

    # Loop through directories
    rows = []
    root_dir = Path(directory)
    for domain_dir in iter_subdirs(root_dir):
        dom = domain_dir.name
        if incl_domain is not None and dom not in incl_domain:
            continue
        for alg_dir in iter_subdirs(domain_dir):
            alg = alg_dir.name
            if incl_algorithm is not None and alg not in incl_algorithm:
                continue
            for run_dir in iter_subdirs(alg_dir):
                cfg_path = run_dir/'config.json'
                log_path = run_dir/'log.txt'
                status_path = run_dir/'status.txt'
                if not(cfg_path.is_file() and log_path.is_file() and status_path.is_file()):
                    continue

                run_dt, run_id = parse_run(run_dir.name)

                # Filter by ID
                if incl_ids is not None and run_id not in incl_ids:
                    continue

                # Filter by recency
                if days is not None:
                    now = datetime.now()
                    diff = now - run_dt
                    if diff.days > days:
                        continue

                # Filter by status
                with status_path.open('r') as f:
                    status = f.read(10).split(' ')[0]
                    if incl_status is not None and status not in incl_status:
                        continue

                # Filter by hyperparameters
                if len(hp_filters) > 0 and not filter_check(run_dir, hp_filters):
                    continue

                # Passed all filters
                if should_plot:
                    tfevents_files = [f for f in run_dir.iterdir() if 'tfevents' in f.name]
                    assert len(tfevents_files) == 1
                    plot_values = read_tfevents(tfevents_files[0], plot_keys)
                    if all(len(v) == 0 for v in plot_values):
                        continue
                else:
                    plot_values = None
                rows.append([
                    status, run_dt, run_id, dom, alg, cfg_path, log_path, plot_values
                ])

    # Sort by date/time
    rows = sorted(rows, reverse=True)

    # Create plot (if applicable)
    if should_plot:
        for i, key in enumerate(plot_keys):
            plot_path = root_dir/f'plot-{i}.png'
            plt.figure()
            plt.ylabel(key)
            for _, _, run_id, _, _, _, _, plot_values in reversed(rows):
                xs = [x for x, _ in plot_values[key]]
                ys = [y for _, y in plot_values[key]]
                plt.plot(xs, ys, label=run_id)
            plt.legend()
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

    # Write HTML
    doc, tag, text, line = Doc().ttl()
    with tag('head'):
        with tag('style'):
            text("th, td { padding-left: 5px; padding-right: 5px }")
            text("tr { border-bottom: 1px }")
    with tag('body'):
        if should_plot:
            for i, _ in enumerate(plot_keys):
                doc.stag('img', src=f'plot-{i}.png')

        with tag('table'):
            # Header
            with tag('tr'):
                line('th', 'Status')
                line('th', 'Start Date')
                line('th', 'Start Time')
                line('th', 'ID')
                line('th', 'Domain')
                line('th', 'Algorithm')
                line('th', 'Config')
                line('th', 'Log')

            # Rows
            for status, run_dt, run_id, dom, alg, cfg_path, log_path, _ in rows:
                with tag('tr'):
                    # Basic info
                    line('td', status)
                    line('td', f'{run_dt.year}/{run_dt.month:02}/{run_dt.day:02}')
                    line('td', f'{run_dt.hour:02}:{run_dt.minute:02}:{run_dt.second:02}')
                    line('td', run_id)
                    line('td', dom)
                    line('td', alg)

                    # Links to important files
                    with tag('td'):
                        with tag('a', href=relative_path(root_dir, cfg_path), target='_blank'):
                            text('config')
                    with tag('td'):
                        with tag('a', href=relative_path(root_dir, log_path), target='_blank'):
                            text('log')

    return doc.getvalue()


class RequestHandler(SimpleHTTPRequestHandler):
    def _parse_path(self):
        split = self.path.split('?')
        if len(split) == 1:
            # No queries
            return self.path, {}
        elif len(split) == 2:
            path, query = split
            query = dict(parse_qsl(query))
            return path, query
        else:
            raise RuntimeError('Too many ? in URL')

    def _send_html(self, s):
        # Encode
        enc = sys.getfilesystemencoding()
        encoded = s.encode(enc)

        # Create tempfile
        f = TemporaryFile('w+b')
        f.write(encoded)
        f.seek(0)

        # Respond with headers
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "text/html; charset=%s" % enc)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()

        # Return file, which will be copied as response
        return f

    def send_head(self):
        path, kwargs = self._parse_path()
        if path == '/':
            return self._send_html(index_page(self.directory, **kwargs))
        else:
            return super().send_head()


def launch_browser(ip, port):
    import webbrowser
    import time
    print(f'Opening browser...')
    time.sleep(0.25) # give the server a bit of time to start up
    webbrowser.open_new_tab(f'{ip}:{port}')


def main(directory, ip, port, open_browser):
    if open_browser:
        launch_proc = mp.Process(target=launch_browser, args=(port,))
        launch_proc.start()
    else:
        launch_proc = None

    server_address = (ip, port)
    Handler = functools.partial(RequestHandler, directory=directory)
    with HTTPServer(server_address, Handler) as httpd:
        print(f'Server listening on port {port}...')
        httpd.serve_forever()

    if launch_proc is not None:
        print('Joining launch process...')
        launch_proc.join()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--directory', required=True)
    parser.add_argument('--ip', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=1337)
    parser.add_argument('-o', '--open-browser', action='store_true')
    main(**vars(parser.parse_args()))