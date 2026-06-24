from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Component:
    name: str
    kind: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "details": self.details,
        }


@dataclass
class Layer:
    index: int
    name: str
    layer_type: str
    components: List[Component] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "type": self.layer_type,
            "components": [component.to_dict() for component in self.components],
        }


@dataclass
class Architecture:
    model_id: str
    revision: str
    model_type: str
    architectures: List[str]
    summary: Dict[str, Any]
    text_decoder: Dict[str, Any]
    vision_encoder: Optional[Dict[str, Any]]
    layers: List[Layer]
    files: List[str] = field(default_factory=list)
    skipped_weight_files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "model_type": self.model_type,
            "architectures": self.architectures,
            "summary": self.summary,
            "text_decoder": self.text_decoder,
            "vision_encoder": self.vision_encoder,
            "layers": [layer.to_dict() for layer in self.layers],
            "files": self.files,
            "skipped_weight_files": self.skipped_weight_files,
            "notes": self.notes,
        }
