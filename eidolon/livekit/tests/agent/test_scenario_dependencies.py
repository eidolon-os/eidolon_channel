def test_input_pipelines_do_not_import_conversation_scenarios():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "agent"
    for directory in ("half_duplex", "full_duplex", "shared"):
        for path in (root / directory).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert "coordination" not in (node.module or "").split("."), path
                elif isinstance(node, ast.Import):
                    assert all("coordination" not in item.name.split(".") for item in node.names), (
                        path
                    )
