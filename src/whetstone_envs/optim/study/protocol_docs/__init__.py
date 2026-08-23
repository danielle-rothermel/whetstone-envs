"""The pre-registration documents this package's protocols are registered on.

A pre-registration that only exists in one person's home directory is not
verifiable by anyone else: ``init`` hashes the document at
:data:`~whetstone_envs.optim.study.protocols.PROTOCOL_DOC_PATH` and refuses
to author a study without it, so a checkout that could not read that path
could not initialise a study at all.

The registered text therefore lives here, in the package, byte-identical to
the durable copy it was written in. It ships in the wheel, so the digest a
manifest records is checkable from any checkout and from an installed
package -- which is what makes the recorded digest evidence rather than a
note about a file someone else can see.

The durable copy remains the authoring original. This is a versioned copy
of it, not a fork: they are the same bytes, and a golden test pins the
digest so they cannot drift apart silently.
"""

from __future__ import annotations

__all__ = ["STEP10_C19_PROTOCOL_DOC"]

#: The Step 10 c19 pre-registration, by filename within this package.
STEP10_C19_PROTOCOL_DOC = "step10-c19-protocol.md"
