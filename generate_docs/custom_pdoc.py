#!usr/bin/env python3
"""pdoc's CLI interface and helper functions."""

## USE THE FOLLOWING COMMAND:
## python3 ./custom_pdoc.py voltrace -o docs  --force --html --config latex_math=True 

import argparse
import ast
import os
import os.path as path
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence
from warnings import warn

import pdoc
from pdoc import _get_config
from pdoc import tpl_lookup

parser = argparse.ArgumentParser(
    description="Automatically generate API docs for Python modules.",
    epilog="Further documentation is available at <https://pdoc3.github.io/pdoc/doc>.",
)
aa = parser.add_argument
mode_aa = parser.add_mutually_exclusive_group().add_argument

aa(
    '--version', action='version', version=f'%(prog)s {pdoc.__version__}')
aa(
    "modules",
    type=str,
    metavar='MODULE',
    nargs="+",
    help="The Python module name. This may be an import path resolvable in "
    "the current environment, or a file path to a Python module or "
    "package.",
)
aa(
    "-c", "--config",
    type=str,
    metavar='OPTION=VALUE',
    action='append',
    default=[],
    help="Override template options. This is an alternative to using "
         "a custom config.mako file in --template-dir. This option "
         "can be specified multiple times.",
)
aa(
    "--filter",
    type=str,
    metavar='STRING',
    default=None,
    help="Comma-separated list of filters. When specified, "
         "only identifiers containing the specified string "
         "will be shown in the output. Search is case sensitive. "
         "Has no effect when --http is set.",
)
aa(
    "-f", "--force",
    action="store_true",
    help="Overwrite any existing generated (--output-dir) files.",
)
mode_aa(
    "--html",
    action="store_true",
    help="When set, the output will be HTML formatted.",
)
aa(
    "--html-dir",
    type=str,
    help=argparse.SUPPRESS,
)
aa(
    "-o", "--output-dir",
    type=str,
    metavar='DIR',
    help="The directory to output generated HTML/markdown files to "
         "(default: ./html for --html).",
)
aa(
    "--html-no-source",
    action="store_true",
    help=argparse.SUPPRESS,
)
aa(
    "--overwrite",
    action="store_true",
    help=argparse.SUPPRESS,
)
aa(
    "--external-links",
    action="store_true",
    help=argparse.SUPPRESS,
)
aa(
    "--link-prefix",
    type=str,
    help=argparse.SUPPRESS,
)
aa(
    "--close-stdin",
    action="store_true",
    help="When set, stdin will be closed before importing, to account for "
         "ill-behaved modules that block on stdin."
)

aa(
    "--skip-errors",
    action="store_true",
    help="Upon unimportable modules, warn instead of raising."
)

args = argparse.Namespace()


def module_path(m: pdoc.Module, ext: str):
    return path.join(args.output_dir, *re.sub(r'\.html$', ext, m.url()).split('/'))


def write_module_html(module, page_content=None, **kwargs):
    config = _get_config(**kwargs)
    t = tpl_lookup.get_template('/html.mako')
    return t.render(module=module, page_content=page_content, **config).strip()


def recursive_write_files(m: pdoc.Module, ext: str, pages, **kwargs):
    assert ext in ('.html', '.md')
    filepath = module_path(m, ext=ext)

    dirpath = path.dirname(filepath)
    if not os.access(dirpath, os.R_OK):
        os.makedirs(dirpath)

    with open(filepath, 'w', encoding='utf-8') as f:
        print(filepath)
        f.write(write_module_html(m, pages=pages, **kwargs))
    
    for submodule in m.submodules():
        recursive_write_files(submodule, pages=pages, ext=ext, **kwargs)

def write_page(m: pdoc.Module, file_in, **kwargs):

    with open('pages/' + file_in, 'r', encoding='utf-8') as file:
        file_content = file.read()
     
    output_file = path.join(args.output_dir, 'voltrace', file_in.replace('.md', '.html'))
        
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(write_module_html(m, page_content=file_content, **kwargs))

    print('Written page: ' + output_file)
    
def _warn_deprecated(option, alternative='', use_config_mako=False):
    msg = f'Program option `{option}` is deprecated.'
    if alternative:
        msg += f' Use `{alternative}`'
        if use_config_mako:
            msg += ' or override config.mako template'
        msg += '.'
    warn(msg, DeprecationWarning, stacklevel=2)


def main(_args=None):
    """ Command-line entry point """
    global args
    args = _args or parser.parse_args()

    if args.close_stdin:
        sys.stdin.close()

    if (args.html or args.http) and not args.output_dir:
        args.output_dir = 'html'

    if args.html_dir:
        _warn_deprecated('--html-dir', '--output-dir')
        args.output_dir = args.html_dir
    if args.overwrite:
        _warn_deprecated('--overwrite', '--force')
        args.force = args.overwrite

    template_config = {}
    for config_str in args.config:
        try:
            key, value = config_str.split('=', 1)
            value = ast.literal_eval(value)
            template_config[key] = value
        except Exception:
            raise ValueError(
                f'Error evaluating --config statement "{config_str}". '
                'Make sure string values are quoted?'
            )

    if args.html_no_source:
        _warn_deprecated('--html-no-source', '-c show_source_code=False', True)
        template_config['show_source_code'] = False
    if args.link_prefix:
        _warn_deprecated('--link-prefix', '-c link_prefix="foo"', True)
        template_config['link_prefix'] = args.link_prefix
    if args.external_links:
        _warn_deprecated('--external-links')
        template_config['external_links'] = True

    assert path.isdir('templates')
    pdoc.tpl_lookup.directories.insert(0, 'templates')

    # Support loading modules specified as python paths relative to cwd
    sys.path.append(os.getcwd())

    from glob import glob
    from sysconfig import get_path
    libdir = get_path("platlib")
    sys.path.append(libdir)

    if args.filter and args.filter.strip():
        def docfilter(obj, _filters=args.filter.strip().split(',')):
            return any(f in obj.refname or
                       isinstance(obj, pdoc.Class) and f in obj.doc
                       for f in _filters)
    else:
        docfilter = None

    voltrace_module = pdoc.Module('voltrace', docfilter=docfilter, skip_errors=args.skip_errors)
    voltrace_pro_module = pdoc.Module('voltrace_pro', docfilter=docfilter, skip_errors=args.skip_errors)

    modules = [voltrace_module, voltrace_pro_module]
    
    pdoc.link_inheritance()

    # Loading is done. Output stage ...
    config = pdoc._get_config(**template_config)

    # Load configured global markdown extensions
    # XXX: This is hereby enabled only for CLI usage as for
    #  API use I couldn't figure out where reliably to put it.
    if config.get('md_extensions'):
        from pdoc.html_helpers import _md
        _kwargs = {'extensions': [], 'configs': {}}
        _kwargs.update(config.get('md_extensions', {}))
        _md.registerExtensions(**_kwargs)

    pages = {
        'einzel-lens.md': 'Einzel lens',
    }
    
    for module in modules:
        recursive_write_files(module, pages=pages, ext='.html', **template_config, voltrace_module=voltrace_module, voltrace_pro_module=voltrace_pro_module)

    for p in pages.keys():
        write_page(voltrace_module, p, **template_config, pages=pages, voltrace_module=voltrace_module, voltrace_pro_module=voltrace_pro_module)

main(parser.parse_args())
