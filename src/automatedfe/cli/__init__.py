"""Package boundary for command-line adapters.

Command implementations remain in ``scripts`` until phase 3.  Keeping this
initializer intentionally side-effect free makes ``automatedfe.cli`` safe to
import before the unified dispatcher is introduced.
"""

__all__: list[str] = []
