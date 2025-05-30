import argparse
import fnmatch
import tqdm
import re
import subprocess
import stat
import sys, os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=Path, required=True)
parser.add_argument('-l', '--lib_paths', nargs="*")
parser.add_argument('-o', '--out_dir', type=Path, default=None, help="optional directory for copying dylibs")
parser.add_argument('--safe', action="store_true", help="safe mode")
parser.add_argument('--sign_id', type=str, default="-", help="codesign sign")
parser.add_argument('--noclean_rpath', action="store_true", help="does not clean rpath")
parser.add_argument('--verbose', action="store_true", help="verbose output")
args = parser.parse_args()


DEFAULT_EXCLUDE_PATHS_SOLVE = ['/System/Library/Frameworks', '@executable_path', '/usr/lib']
DEFAULT_EXCLUDE_LIBS = [r'libc\+\+.*', r'libSystem\.B\.*', r'libobjc\.A\.*']


def is_executable(file_path):
    # Check if the file exists and is a regular file
    if not os.path.isfile(file_path):
        return False
    
    try:
        result = subprocess.run(
            ['file', '--brief', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        # Check if the file is considered a Mach-O dynamically linked shared library or executable
        if 'Mach-O' in output and ('executable' in output or 'dynamically linked shared library' in output or 'bundle' in output):
            return True
    except subprocess.CalledProcessError:
        pass

    return False


def extract_rpaths(executable_path, replace_dynamic_paths = True):
    # Run otool to get load commands
    try:
        result = subprocess.run(['otool', '-l', executable_path], 
                                check=True, 
                                text=True, 
                                capture_output=True)
        output = result.stdout

        # Regex pattern to capture RPATH entries
        rpath_pattern = re.compile(r'LC_RPATH.*?\n.*?path (.*?) \(offset', re.S)
        
        # Find all RPATH entries
        rpaths = rpath_pattern.findall(output)

        if replace_dynamic_paths:
            for i, r in enumerate(rpaths):
                rpaths[i] = re.sub("@loader_path", str(executable_path.parent), r)
        

        for i, r in enumerate(rpaths):
            if r.startswith("@loader_path") or r.startswith("@loader_path") or r.startswith("@executable_path"):
                continue 
            rpaths[i] = Path(r)
        return rpaths

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []


def get_library_name(libname):
    return os.path.splitext(os.path.basename(libname))[0].split('.')[0]


def get_dependencies(filepath, dir_filter=[], lib_filter=[], rpath=None):
    if not os.path.exists(filepath): return None
    rpath = rpath or extract_rpaths(filepath)
    try:
        result = subprocess.run(['otool', '-L', filepath], 
                                check=True, 
                                text=True, 
                                capture_output=True)
        # Return the output
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"
    result_lines = result.stdout.split('\n')
    deps = {}
    for r in result_lines:
        result_parsed = re.match(r"\s*(.+)\s+\((.+)\)$", r)
        if result_parsed is None: 
            continue
        outs = result_parsed.groups()
        path = outs[0]
        dep_name = path.split('/')[-1]
        original_path = str(path)
        for rp in rpath:
            path_tmp = Path(str(path).replace('@rpath', str(rp)))
            if os.path.exists(path_tmp):
                path = path_tmp
                break
        if True in [Path(path).is_relative_to(r) for r in dir_filter]: continue
        if True in [re.search(f, str(path)) is not None for f in lib_filter]: continue
        deps[get_library_name(dep_name)] = {'path': path, 'version': outs[1], 'origin': filepath, 'linked': original_path}
    return deps


def get_architectures(file_path):
    try:
        result = subprocess.run(
            ['file', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        return output.split(" ")[-1]
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"



def get_find_results(directory, pattern):
    try:
        # Run the find command
        command = ['find', directory, '-name', pattern]
        print(" ".join(command))
        result = subprocess.run(command, 
                                check=True, 
                                text=True, 
                                capture_output=True)
        file_list = result.stdout.strip().split('\n')
        file_list = [f for f in file_list if f]
        return file_list

    except subprocess.CalledProcessError as e:
        return []


def find_candidates_for(lib_name, lib_dir, lib_arch, allow_different_arch: bool = True):
    lib_name, lib_ext = os.path.splitext(lib_name)
    lib_name_parts = lib_name.split('.')
    candidates = []
    for l in lib_dir: 
        print('parsing %s'%lib_name_parts)
        for i in reversed(range(1, len(lib_name_parts)+1)):
            results = get_find_results(l, ".".join(lib_name_parts[:i]) + "*" + lib_ext)
            if len(results) > 0:
                break
        candidates.extend(results)
    print("candidates before filtering : ", candidates)
    if len(candidates) == 0: 
        return [] 
    candidates_filt_arch = list(filter(lambda x: get_architectures(x) == lib_arch, candidates))
    if len(candidates_filt_arch) == 0:
        print('[Warning] Candidates found for %s, but with wrong architecture'%lib_name)
        if not allow_different_arch:
            return [] 
    else:
        candidates = candidates_filt_arch
    print(candidates)

    for i, c in enumerate(candidates):
        while os.path.islink(c):
            c = os.readlink(os.path.abspath(c))
            if not os.path.isabs(c):
                c = os.path.join(os.path.dirname(candidates[i]), c)
            candidates[i] = c
    return list(set(candidates))


def find_most_relevant_dylib_candidate(name, candidates, accept_weak_matching=False):
    name_ext = os.path.splitext(name)[1][1:]
    re_result = re.match(rf"^([\d\w_]+)\.?([\d\.]*)\.{name_ext}$", name)
    if re_result is None: 
        print(f'[Warning] Could not parse name {name} for dynamic lib retrieving ; taking first : {candidates[0]}')
        return candidates[0]
    lib_name, lib_version = re_result.groups()
    matching_name_and_version = []
    matching_name = []
    matching = []
    for c in candidates:
        c_name = os.path.basename(c)
        re_result = re.match(rf"^([\d\w_]+)\.?([\d\.]*)\.{name_ext}$", c_name)
        if re_result is not None:
            c_name, c_version = re_result.groups()
            if lib_name == c_name and c_version.startswith(lib_version):
                matching_name_and_version.append(c)
            elif lib_name == c_name: 
                matching_name.append(c)
            else:
                matching.append(c)
    if len(matching_name_and_version) > 0:
        return matching_name_and_version[0]
    elif len(matching_name) > 0: 
        print(f'[Warning] Could only find candidate with different version : {matching_name[0]}')
        return matching_name[0]
    else: 
        if accept_weak_matching:
            print(f'[Warning] Could only find weakly matching library : {matching[0]}')
            return matching[0]
        else:
            return None


def clean_rpath_for_target_executable(lib_path): 
    actions = []
    rpaths = extract_rpaths(lib_path, replace_dynamic_paths=False)
    for p in rpaths:
        if not str(p).startswith('@loader_path'): actions.append(['-delete_rpath', p, lib_path])
    return actions


def most_relevant_lib(lib_name, path_dicts, dep_paths=[], arch="arm64"):
    path_list = [Path(p['path']) for p in path_dicts]
    path_filtered_list = []
    for p in path_list:
        print('looking in %s...'%p)
        while os.path.islink(p):
            p = Path(p.parent / os.readlink(p)).resolve().absolute()
        if not p.exists(): continue
        if not is_executable(p): continue
        path_filtered_list.append(p)
    path_list = list(set(path_filtered_list))

    path_list = filter(lambda x: x.exists(), path_list)
    path_list = filter(lambda x: is_executable(x), path_list)
    path_list = list(path_list)
    if len(path_list) == 0:
        # not found; look for candidate
        candidates = find_candidates_for(f"{lib_name}.dylib", dep_paths, arch)
        candidate = find_most_relevant_dylib_candidate(f"{lib_name}.dylib", candidates)
        if candidate is None: 
            raise RuntimeError('no valid library found for %s in %s (candidates : %s)'%(lib_name, dep_paths, candidates))
        return candidate
    # find in priority the librairies given in arguments
    for libdir in map(Path, dep_paths):
        for path in path_list:
            if path.is_relative_to(libdir): return path
    return path_list[0]

def lib_from_exc_path(exc_path, lib_path):
    exc_parts = exc_path.parts
    lib_parts = lib_path.parts
    assert exc_parts[0] == lib_parts[0], "files do not have a single common root"
    i = 0
    while exc_parts[i] == lib_parts[i]:
        i+=1
    return os.path.join(*([".."] * (len(exc_parts[i:])-1) + list(lib_parts[i:])))

def parse_actions_from_executable(exec_path, dep_paths=[], main_dir = None, verbose=False):
    if not os.path.exists(exec_path):
        raise FileNotFoundError(exec_path)
    if verbose: print(f'parsing {str(exec_path)}...')
    libs_deps = {}
    libs_paths = {}
    libs_analysed = []
    libs_to_analyse = [exec_path]
    libs_hash = {}
    libs_hash_linked = {}

    arch = get_architectures(exec_path)
    # analyse dependencies
    while len(libs_to_analyse) != 0:
        if verbose: print('anlysing librairies %s'%libs_to_analyse)
        parsed_deps = []

        # fetch dependencies
        for current_lib in libs_to_analyse:
            if verbose: print('fetching dependencies from %s'%(current_lib))
            lib_deps = get_dependencies(current_lib, dir_filter=DEFAULT_EXCLUDE_PATHS_SOLVE, lib_filter=DEFAULT_EXCLUDE_LIBS)
            for dep_name, dep_params in lib_deps.items():
                parsed_deps.append(dep_name)
                libs_deps[dep_name] = libs_deps.get(dep_name, []) + [dep_params]
                libs_hash[get_library_name(dep_params['path'])] = libs_hash.get(get_library_name(dep_params['path']), []) + [dep_params['origin']]
                libs_hash_linked[get_library_name(dep_params['path'])] = libs_hash_linked.get(get_library_name(dep_params['path']), []) + [dep_params['linked']]
            del libs_to_analyse[libs_to_analyse.index(current_lib)]
            libs_analysed.append(get_library_name(current_lib))
         
        # parse dependencies
        for p in parsed_deps:
            if verbose: print('fetching paths for %s'%p)
            if p not in libs_paths:
                lib_path = most_relevant_lib(p, libs_deps[p], dep_paths, arch)
                print('found lib : %s'%lib_path)
                libs_paths[p] = Path(lib_path)
            if p not in libs_analysed:
                print('adding %s for parsing'%p)
                libs_to_analyse.append(libs_paths[p])

    actions = []
    exec_dir = exec_path.parent
    os.makedirs(str(main_dir.resolve()), exist_ok=True)
    for k, v in libs_paths.items():
        if not (main_dir / v.name).resolve().exists():
            actions.append(['copy', str(v), str(main_dir)])
    for k, v in libs_hash.items():
        for i, v_tmp in enumerate(v):
            # if str(libs_hash_linked[k][i]).startswith('@rpath'):
            #     continue
            if get_library_name(libs_hash_linked[k][i]) == get_library_name(v_tmp.name):
                if v_tmp.stem == exec_path.stem:
                    actions.append(['-id', f"@loader_path/{v_tmp.name}", str(exec_dir / v_tmp.name)])
                else:
                    actions.append(['-id', f"@loader_path/{v_tmp.name}", str(main_dir / v_tmp.name)])
            else:
                current_dep = str(libs_hash_linked[k][i]) 
                # new_dep = f"@loader_path/{libs_paths[get_library_name(current_dep)].name}"
                if v_tmp.stem == exec_path.stem:
                    new_dep = f"@loader_path/{lib_from_exc_path(exec_path, main_dir / libs_paths[get_library_name(current_dep)].name)}"
                    actions.append(['-change', current_dep, new_dep, str(exec_dir / v_tmp.name)])
                else:
                    new_dep = f"@loader_path/{libs_paths[get_library_name(current_dep)].name}"
                    actions.append(['-change', current_dep, new_dep, str(main_dir / v_tmp.name)])
    return actions


def perform_action(action, main_dir):
    try:
        if action[0] == "-change": 
            # Run the find command
            result = subprocess.run(['install_name_tool', 
                                     '-change', 
                                     action[1], 
                                     action[2], 
                                     action[3]],
                                    check=True, 
                                    text=True, 
                                    capture_output=False)
            return 
        elif action[0] == "-id":
            target_path = str(main_dir / os.path.split(action[1])[-1]) 
            result = subprocess.run(['install_name_tool', 
                                     '-id', 
                                     action[1], 
                                     target_path],
                                    check=True, 
                                    text=True, 
                                    capture_output=False)
        elif action[0] == "copy":
            target_path = main_dir / os.path.split(action[1])[-1]
            if not target_path.exists(): 
                result = subprocess.run(['cp',
                                    action[1],
                                    str(target_path)],  
                                    check=True, 
                                    text=True, 
                                    capture_output=False)
        elif action[0] == "-delete_rpath":
            target_path = str(main_dir / os.path.split(action[2])[-1]) 
            result = subprocess.run(['install_name_tool', 
                                     '-delete_rpath', 
                                     action[1], 
                                     target_path], 
                                    check=True, 
                                    text=True, 
                                    capture_output=False)
        elif action[0] == "-add_rpath":
            target_path = str(main_dir / os.path.split(action[2])[-1]) 
            result = subprocess.run(['install_name_tool', 
                                     '-add_rpath', 
                                     action[1], 
                                     target_path], 
                                    check=True, 
                                    text=True, 
                                    capture_outpu=False)
        elif action[0] == "clean_rpath":
            rpaths = extract_rpaths(str(main_dir / action[1]), replace_dynamic_paths=False)
            for r in rpaths: 
                try:
                    subprocess.run(['install_name_tool', '-delete_rpath', str(r), str(action[1])])
                except subprocess.SubprocessError as e: 
                    print("problem with clean_rpath : %s"%e)
                    pass

        else:
            print('[Warning] not known action : %s'%action) 

    except subprocess.CalledProcessError as e: 
        raise e


def print_action(idx, action):
    if action[0] == "copy":
        print(f'{idx:02d} cp {"|".join(action[1:])} -> @rpath')
    elif action[0] == "-change": 
        print(f"{idx:02d} -change: {os.path.basename(action[3])}: {action[1]}->{action[2]}")
    elif action[0] == "-id":
        print(f"{idx:02d} -id: {os.path.basename(action[1])}")
    elif action[0] == "-delete_rpath":
        print(f"{idx:02d} -delete_rpath: {action[1]} -> {action[2]}")
    elif action[0] == "clean_rpath": 
        print(f"{idx:02d} clean_rpath: {action[1]}")
    else:
        print(idx, action)


if __name__ == "__main__":
    exec_path = Path(args.path)
    main_dir = args.out_dir or args.path.parent 
    actions = parse_actions_from_executable(exec_path, args.lib_paths, main_dir, args.verbose)


    if args.safe:
        for i, a in enumerate(actions):
            print_action(i, a)
        print('continue?')
        answ = ""
        while answ.lower() not in ['y', 'n']:
            answ = input('[y/n] : ')
        if answ.lower() == "n": raise SystemExit("raised by user") 

    os.makedirs(str(main_dir.resolve()), exist_ok=True)
    for a in tqdm.tqdm(actions): 
        perform_action(a, main_dir = main_dir)

    if not args.noclean_rpath:
        for m in [os.path.join(main_dir, m) for m in os.listdir(main_dir)] + [args.path]:
            if is_executable(m):
                perform_action(['clean_rpath', m], main_dir)


    if args.sign_id == "": args.sign_id = "-"
    for m in [os.path.join(main_dir, m) for m in os.listdir(main_dir)] + [args.path]:
        try:
            subprocess.run(['chmod', '+x', m])
            subprocess.run(['codesign', '--deep', '--force', '--options=runtime', '--sign', args.sign_id, m])
            subprocess.run(["xattr", "-r", "-d", "com.apple.quarantine", m])
        except subprocess.CalledProcessError as e: 
            print(f'Could not chmod / codesign file {m} ; codesign needed')
                