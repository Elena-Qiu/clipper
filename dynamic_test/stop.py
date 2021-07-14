from __future__ import print_function
from clipper_admin import ClipperConnection, DockerContainerManager


clipper_conn = ClipperConnection(DockerContainerManager(use_centralized_log=False))
clipper_conn.stop_all()