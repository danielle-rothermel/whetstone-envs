from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from pydantic import ValidationError

from whetstone_envs.instances import (
    Instance,
    InstanceGroup,
    InstancePool,
    InstancePoolSplit,
)
from whetstone_envs.manifests import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    content_hash,
)


def pool() -> InstancePool:
    return InstancePool(
        instance_group=InstanceGroup(
            name="generic",
            instances=tuple(
                Instance(
                    id=f"sample-{index}",
                    seed=index,
                    categories=("common", "rare")
                    if index == 0
                    else ("common",),
                    prompt_inputs={"value": str(index)},
                    gold=str(index),
                )
                for index in range(6)
            ),
        )
    )


class InstanceTests(TestCase):
    def test_inputs_are_detached_and_frozen(self) -> None:
        inputs = {"value": "one"}
        instance = Instance(
            id="one",
            seed=1,
            categories=("a",),
            prompt_inputs=inputs,
            gold="one",
        )
        inputs["value"] = "changed"
        self.assertEqual(instance.prompt_inputs["value"], "one")
        with self.assertRaises(TypeError):
            instance.prompt_inputs["value"] = "changed"  # ty: ignore[invalid-assignment]
        with self.assertRaises(ValidationError):
            instance.seed = 2  # ty: ignore[invalid-assignment]
        with self.assertRaises(ValidationError):
            Instance(
                id="one",
                seed=True,
                categories=("a",),
                prompt_inputs=inputs,
                gold="one",
            )

    def test_named_splits_are_deterministic_and_source_linked(self) -> None:
        source = pool()
        result = source.split({"selection": 2, "reserve": 3})
        self.assertEqual(result, source.split({"selection": 2, "reserve": 3}))
        self.assertIs(result.pool, source)
        self.assertEqual(tuple(result.splits), ("selection", "reserve"))
        ids = []
        for name, size in (("selection", 2), ("reserve", 3)):
            group = result.splits[name]
            self.assertEqual(len(group.instances), size)
            self.assertEqual(
                group.instances,
                tuple(
                    i
                    for i in source.instance_group.instances
                    if i in group.instances
                ),
            )
            ids.extend(i.id for i in group.instances)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 5)
        with self.assertRaises(TypeError):
            result.splits["extra"] = result.split_groups[0]  # ty: ignore[invalid-assignment]

    def test_split_rejects_foreign_instances_and_oversubscription(
        self,
    ) -> None:
        source = pool()
        foreign = Instance(
            id="foreign",
            seed=10,
            categories=("other",),
            prompt_inputs={"value": "foreign"},
            gold="foreign",
        )
        with self.assertRaises(ValidationError):
            InstancePoolSplit(
                pool=source,
                split_groups=(InstanceGroup(name="x", instances=(foreign,)),),
            )
        for sizes in ({"x": 7}, {"x": True}, {" ": 1}, {"x": -1}):
            with (
                self.subTest(sizes=sizes),
                self.assertRaises((ValueError, TypeError)),
            ):
                source.split(sizes)

    def test_pool_rejects_duplicate_public_inputs(self) -> None:
        original = pool().instance_group.instances[0]
        duplicate = original.model_copy(update={"id": "another", "seed": 99})
        with self.assertRaisesRegex(
            ValidationError, "duplicate public prompt identity"
        ):
            InstancePool(
                instance_group=InstanceGroup(
                    name="x", instances=(original, duplicate)
                )
            )


class ManifestTests(TestCase):
    def test_identity_and_persisted_keys_are_pinned(self) -> None:
        manifest = Manifest.from_pool(
            pool(), generator_version="fixture/v1", seed_range=(0, 6)
        )
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 2)
        self.assertEqual(
            set(manifest.to_dict()),
            {
                "schema_version",
                "generator_version",
                "seed_range",
                "category_counts",
                "content_hash",
            },
        )
        self.assertEqual(
            str(content_hash(pool())),
            "f4dfb3fb380a4ecfe50185d97dffd41a691580a6aa3ceea49ee8b9e1615a8afc",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.write(path)
            self.assertEqual(path.read_text(), manifest.to_json())
            self.assertEqual(Manifest.read(path), manifest)
            self.assertTrue(Manifest.read(path).matches_pool(pool()))

    def test_manifest_rejects_schema_drift_and_out_of_range_seeds(
        self,
    ) -> None:
        manifest = Manifest.from_pool(
            pool(), generator_version="fixture/v1", seed_range=(0, 6)
        )
        for change in (
            {"schema_version": 1},
            {"schema_version": True},
            {"unknown": "field"},
            {"category_counts": {"common": 0}},
        ):
            with (
                self.subTest(change=change),
                self.assertRaises((ValueError, TypeError)),
            ):
                Manifest.from_dict(manifest.to_dict() | change)
        with self.assertRaisesRegex(ValueError, "outside declared"):
            Manifest.from_pool(
                pool(), generator_version="fixture/v1", seed_range=(0, 5)
            )
