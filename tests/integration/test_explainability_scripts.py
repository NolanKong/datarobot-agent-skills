# Copyright (c) 2026 DataRobot, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for datarobot-model-explainability scripts and insights SDK patterns."""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from datarobot.insights import (
    ConfusionMatrix,
    LiftChart,
    RocCurve,
    ShapImpact,
    ShapMatrix,
    ShapPreview,
)
from datarobot.models.anomaly_assessment import AnomalyAssessmentRecord
from datarobot.models.model import Model

REPO_ROOT = Path(__file__).parent.parent.parent
COMPUTE_SHAP_MATRIX_PATH = (
    REPO_ROOT / "skills/datarobot-model-explainability/scripts/compute_shap_matrix.py"
)

FAKE_SHAP_MATRIX_PAYLOAD = {
    "id": "insight-1",
    "entity_id": "model-abc",
    "project_id": "proj-1",
    "source": "validation",
    "data": {
        "index": [0, 1],
        "link_function": "identity",
        "base_value": 0.5,
        "colnames": ["income", "age"],
        "matrix": [[0.1, 0.2], [0.3, 0.4]],
    },
}

FAKE_SHAP_IMPACT_PAYLOAD = {
    "id": "insight-2",
    "entity_id": "model-abc",
    "project_id": "proj-1",
    "source": "training",
    "data": {
        "shap_impacts": [
            ["income", 0.6, 0.12],
            ["age", 0.4, 0.08],
        ],
        "base_value": [0.5],
        "capping": None,
        "link": "identity",
        "row_count": 2,
    },
}

FAKE_SHAP_PREVIEW_PAYLOAD = {
    "id": "insight-3",
    "entity_id": "model-abc",
    "project_id": "proj-1",
    "source": "validation",
    "data": {
        "previews": [
            {
                "row_index": 0,
                "prediction_value": 0.72,
                "preview_values": [
                    {
                        "feature_rank": 1,
                        "feature_name": "income",
                        "feature_value": "85000",
                        "shap_value": 0.18,
                        "has_text_explanations": False,
                        "text_explanations": [],
                    }
                ],
            }
        ],
        "previews_count": 1,
    },
}


def load_compute_shap_matrix_module():
    """Load the skill script as a module (not installed as a package)."""
    name = "compute_shap_matrix"
    spec = importlib.util.spec_from_file_location(name, COMPUTE_SHAP_MATRIX_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compute_shap_matrix():
    return load_compute_shap_matrix_module()


@pytest.fixture
def fake_shap_matrix() -> ShapMatrix:
    return ShapMatrix.from_server_data(FAKE_SHAP_MATRIX_PAYLOAD)


class TestInsightsShapMatrixWithFakeData:
    """SDK patterns used in SKILL.md — no live DataRobot API calls."""

    def test_from_server_data_exposes_matrix_and_columns(
        self, fake_shap_matrix: ShapMatrix
    ) -> None:
        assert fake_shap_matrix.entity_id == "model-abc"
        assert fake_shap_matrix.columns == ["income", "age"]
        assert fake_shap_matrix.base_value == 0.5
        assert fake_shap_matrix.link_function == "identity"
        assert len(fake_shap_matrix.matrix) == 2

    def test_export_via_dataframe_matches_matrix(
        self, fake_shap_matrix: ShapMatrix, tmp_path: Path
    ) -> None:
        out = tmp_path / "shap.csv"
        df = pd.DataFrame(fake_shap_matrix.matrix, columns=fake_shap_matrix.columns)
        df.to_csv(out, index=False)

        loaded = pd.read_csv(out)
        assert list(loaded.columns) == ["income", "age"]
        assert loaded.shape == (2, 2)
        assert loaded.iloc[0]["income"] == pytest.approx(0.1)

    def test_instance_get_as_dataframe_is_not_supported(
        self, fake_shap_matrix: ShapMatrix
    ) -> None:
        with pytest.raises(TypeError, match="entity_id"):
            fake_shap_matrix.get_as_dataframe()

    def test_classmethod_get_as_dataframe_parses_csv(
        self, fake_shap_matrix: ShapMatrix
    ) -> None:
        csv_body = "income,age\n0.1,0.2\n0.3,0.4\n"
        mock_response = MagicMock()
        mock_response.text = csv_body

        with patch.object(ShapMatrix, "_client") as mock_client:
            mock_client.get.return_value = mock_response
            df = ShapMatrix.get_as_dataframe(
                entity_id=fake_shap_matrix.entity_id,
                source=fake_shap_matrix.source,
            )

        assert list(df.columns) == ["income", "age"]
        assert len(df) == 2
        mock_client.get.assert_called_once()


class TestInsightsOtherClassesWithFakeData:
    def test_shap_impact_attributes(self) -> None:
        result = ShapImpact.from_server_data(FAKE_SHAP_IMPACT_PAYLOAD)
        assert len(result.shap_impacts) == 2
        assert result.base_value == [0.5]

    def test_shap_preview_attributes(self) -> None:
        result = ShapPreview.from_server_data(FAKE_SHAP_PREVIEW_PAYLOAD)
        assert result.previews_count == 1
        assert result.previews[0]["prediction_value"] == pytest.approx(0.72)


class TestModelDiagnosticsApiSignatures:
    """Guard SKILL.md diagnostics examples against SDK method names."""

    def test_confusion_chart_not_confusion_matrix(self) -> None:
        assert hasattr(Model, "get_confusion_chart")
        assert not hasattr(Model, "get_confusion_matrix")

    def test_feature_effect_not_partial_dependence_helper(self) -> None:
        assert hasattr(Model, "get_feature_effect")
        assert not hasattr(Model, "get_partial_dependence")

    @pytest.mark.parametrize(
        "insight_cls",
        [RocCurve, LiftChart, ConfusionMatrix],
    )
    def test_insights_diagnostics_support_create(self, insight_cls: type) -> None:
        assert hasattr(insight_cls, "create")
        params = inspect.signature(insight_cls.create).parameters
        assert "entity_id" in params


class TestAnomalyAssessmentApiSignatures:
    """Guard SKILL.md anomaly examples against SDK signature drift."""

    def test_get_latest_explanations_has_no_date_parameters(self) -> None:
        params = inspect.signature(
            AnomalyAssessmentRecord.get_latest_explanations
        ).parameters
        assert set(params) == {"self"}

    def test_get_explanations_accepts_date_range_parameters(self) -> None:
        params = inspect.signature(AnomalyAssessmentRecord.get_explanations).parameters
        assert {"start_date", "end_date", "points_count"}.issubset(params)


class TestComputeShapMatrixScript:
    @patch.object(ShapMatrix, "create")
    def test_export_output_writes_csv(
        self,
        mock_create: MagicMock,
        compute_shap_matrix,
        tmp_path: Path,
    ) -> None:
        mock_create.return_value = ShapMatrix.from_server_data(FAKE_SHAP_MATRIX_PAYLOAD)
        out = tmp_path / "export.csv"

        compute_shap_matrix.compute_shap_matrix(
            model_id="model-abc",
            output_path=str(out),
        )

        assert out.exists()
        loaded = pd.read_csv(out)
        assert list(loaded.columns) == ["income", "age"]
        assert len(loaded) == 2
        mock_create.assert_called_once_with(
            entity_id="model-abc",
            source="validation",
            data_slice_id=None,
            external_dataset_id=None,
            quick_compute=None,
        )

    @patch.object(ShapMatrix, "create")
    def test_full_compute_passes_quick_compute_false(
        self,
        mock_create: MagicMock,
        compute_shap_matrix,
    ) -> None:
        mock_create.return_value = ShapMatrix.from_server_data(FAKE_SHAP_MATRIX_PAYLOAD)
        compute_shap_matrix.compute_shap_matrix(
            model_id="model-abc",
            quick_compute=False,
        )
        mock_create.assert_called_once_with(
            entity_id="model-abc",
            source="validation",
            data_slice_id=None,
            external_dataset_id=None,
            quick_compute=False,
        )

    def test_external_test_set_requires_dataset_path(self, compute_shap_matrix) -> None:
        with pytest.raises(ValueError, match="--dataset-path"):
            compute_shap_matrix.compute_shap_matrix(
                model_id="model-abc",
                source="externalTestSet",
            )

    @patch.object(ShapMatrix, "list")
    def test_list_existing_uses_entity_id(
        self,
        mock_list: MagicMock,
        compute_shap_matrix,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_list.return_value = [ShapMatrix.from_server_data(FAKE_SHAP_MATRIX_PAYLOAD)]

        env = {"DATAROBOT_API_TOKEN": "test-token"}
        with patch.dict(os.environ, env, clear=False):
            with patch.object(compute_shap_matrix.dr, "Client"):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "compute_shap_matrix.py",
                        "--model-id",
                        "model-abc",
                        "--list-existing",
                    ],
                ):
                    compute_shap_matrix.main()

        mock_list.assert_called_once_with(entity_id="model-abc")
        assert "Found 1 existing" in capsys.readouterr().out
