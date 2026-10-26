import hashlib
import pytest
from motiogen.source_cohorts import archive_sources

def test_source_archive_preserves_original_when_source_changes(tmp_path):
    root=tmp_path/'repo';root.mkdir();source=root/'worker.py';source.write_text('original\n')
    digest=hashlib.sha256(source.read_bytes()).hexdigest();destination=tmp_path/'archive'
    archive_sources(root,destination,{'worker.py':digest})
    source.write_text('changed\n')
    with pytest.raises(ValueError,match='changed before archival'):
        archive_sources(root,destination,{'worker.py':digest})
    assert (destination/'worker.py').read_text()=='original\n'

def test_source_archive_refuses_rebinding_existing_evidence(tmp_path):
    root=tmp_path/'repo';root.mkdir();source=root/'worker.py';source.write_text('new version\n')
    destination=tmp_path/'archive';destination.mkdir();(destination/'worker.py').write_text('old evidence\n')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='overwrite frozen source'):
        archive_sources(root,destination,{'worker.py':digest})
    assert (destination/'worker.py').read_text()=='old evidence\n'
