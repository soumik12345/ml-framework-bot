import os
from glob import glob
from git import Repo
from typing import List


def fetch_git_repository(
    repository_local_path: str,
    username: str,
    repo_name: str,
    personal_access_token: str,
) -> None:
    os.makedirs(repository_local_path, exist_ok=True)
    if len(glob(os.path.join(repository_local_path, "*"))) == 0:
        repository_url = f"https://{personal_access_token}:x-oauth-basic@github.com/{username}/{repo_name}"
        repository = Repo.clone_from(repository_url, repository_local_path)
    else:
        repository = Repo(repository_local_path)
    repository.remotes.origin.pull()


def get_all_file_paths(directory: str) -> List[str]:
    file_paths = []

    def recurse_folder(folder):
        for entry in os.scandir(folder):
            if entry.is_dir():
                recurse_folder(entry.path)
            else:
                file_paths.append(entry.path)

    recurse_folder(directory)

    return file_paths