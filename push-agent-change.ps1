$current_branch = git rev-parse --abbrev-ref HEAD

if ($current_branch -eq "main" -or $current_branch -eq "master" -or $current_branch -eq "deploy") {
    $new_branch_name = "modularize-agent-context-logic"
    git checkout -b $new_branch_name
} else {
    $new_branch_name = $current_branch
}

git push -u origin $new_branch_name
