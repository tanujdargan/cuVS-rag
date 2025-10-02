"""
Unit tests for SearchResultAggregator

Tests for safe search result aggregation across multiple GPUs with proper
bounds checking, validation, and error handling.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

from search_result_aggregator import (
    SearchResultAggregator,
    SearchResult,
    AggregatedSearchResult,
    SearchConfig,
    combine_search_results,
    filter_search_results_by_distance
)
from gpu_resource_manager import GPUResourceManager


class TestSearchResult:
    """Test SearchResult data model"""
    
    def test_valid_search_result(self):
        """Test creating a valid SearchResult"""
        distances = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        indices = np.array([[10, 20, 30]], dtype=np.int64)
        
        result = SearchResult(
            distances=distances,
            indices=indices,
            gpu_id=0,
            query_time=0.1,
            k_requested=3,
            k_returned=3
        )
        
        assert result.gpu_id == 0
        assert result.query_time == 0.1
        assert result.k_requested == 3
        assert result.k_returned == 3
        assert np.array_equal(result.distances, distances)
        assert np.array_equal(result.indices, indices)
        
    def test_invalid_gpu_id(self):
        """Test SearchResult with invalid GPU ID"""
        distances = np.array([[1.0, 2.0]], dtype=np.float32)
        indices = np.array([[10, 20]], dtype=np.int64)
        
        with pytest.raises(ValueError, match="gpu_id must be non-negative"):
            SearchResult(
                distances=distances,
                indices=indices,
                gpu_id=-1,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            )
            
    def test_invalid_query_time(self):
        """Test SearchResult with invalid query time"""
        distances = np.array([[1.0, 2.0]], dtype=np.float32)
        indices = np.array([[10, 20]], dtype=np.int64)
        
        with pytest.raises(ValueError, match="query_time must be non-negative"):
            SearchResult(
                distances=distances,
                indices=indices,
                gpu_id=0,
                query_time=-0.1,
                k_requested=2,
                k_returned=2
            )
            
    def test_invalid_k_values(self):
        """Test SearchResult with invalid k values"""
        distances = np.array([[1.0, 2.0]], dtype=np.float32)
        indices = np.array([[10, 20]], dtype=np.int64)
        
        # Invalid k_requested
        with pytest.raises(ValueError, match="k_requested must be positive"):
            SearchResult(
                distances=distances,
                indices=indices,
                gpu_id=0,
                query_time=0.1,
                k_requested=0,
                k_returned=2
            )
            
        # k_returned > k_requested
        with pytest.raises(ValueError, match="k_returned.*cannot exceed k_requested"):
            SearchResult(
                distances=distances,
                indices=indices,
                gpu_id=0,
                query_time=0.1,
                k_requested=1,
                k_returned=2
            )
            
    def test_mismatched_array_shapes(self):
        """Test SearchResult with mismatched array shapes"""
        distances = np.array([[1.0, 2.0]], dtype=np.float32)
        indices = np.array([[10, 20, 30]], dtype=np.int64)  # Different shape
        
        with pytest.raises(ValueError, match="distances shape.*!= indices shape"):
            SearchResult(
                distances=distances,
                indices=indices,
                gpu_id=0,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            )
            
    def test_invalid_array_dimensions(self):
        """Test SearchResult with invalid array dimensions"""
        distances = np.array([1.0, 2.0], dtype=np.float32)  # 1D array
        indices = np.array([10, 20], dtype=np.int64)  # 1D array
        
        with pytest.raises(ValueError, match="distances must be 2D array"):
            SearchResult(
                distances=distances,
                indices=indices,
                gpu_id=0,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            )


class TestAggregatedSearchResult:
    """Test AggregatedSearchResult data model"""
    
    def test_valid_aggregated_result(self):
        """Test creating a valid AggregatedSearchResult"""
        final_distances = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        final_indices = np.array([[10, 20], [30, 40]], dtype=np.int64)
        
        gpu_result = SearchResult(
            distances=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            indices=np.array([[10, 20], [30, 40]], dtype=np.int64),
            gpu_id=0,
            query_time=0.1,
            k_requested=2,
            k_returned=2
        )
        
        result = AggregatedSearchResult(
            final_distances=final_distances,
            final_indices=final_indices,
            total_query_time=0.2,
            gpu_results=[gpu_result],
            k_requested=2,
            k_returned=2,
            num_queries=2
        )
        
        assert result.k_requested == 2
        assert result.k_returned == 2
        assert result.num_queries == 2
        assert result.total_query_time == 0.2
        assert len(result.gpu_results) == 1
        
    def test_invalid_aggregated_result_params(self):
        """Test AggregatedSearchResult with invalid parameters"""
        final_distances = np.array([[1.0, 2.0]], dtype=np.float32)
        final_indices = np.array([[10, 20]], dtype=np.int64)
        
        gpu_result = SearchResult(
            distances=np.array([[1.0, 2.0]], dtype=np.float32),
            indices=np.array([[10, 20]], dtype=np.int64),
            gpu_id=0,
            query_time=0.1,
            k_requested=2,
            k_returned=2
        )
        
        # Invalid k_requested
        with pytest.raises(ValueError, match="k_requested must be positive"):
            AggregatedSearchResult(
                final_distances=final_distances,
                final_indices=final_indices,
                total_query_time=0.2,
                gpu_results=[gpu_result],
                k_requested=0,
                k_returned=2,
                num_queries=1
            )
            
        # Invalid num_queries
        with pytest.raises(ValueError, match="num_queries must be positive"):
            AggregatedSearchResult(
                final_distances=final_distances,
                final_indices=final_indices,
                total_query_time=0.2,
                gpu_results=[gpu_result],
                k_requested=2,
                k_returned=2,
                num_queries=0
            )


class TestSearchConfig:
    """Test SearchConfig data model"""
    
    def test_valid_search_config(self):
        """Test creating a valid SearchConfig"""
        config = SearchConfig(
            k=10,
            search_params={'nprobe': 32},
            parallel_search=True,
            timeout_seconds=30.0,
            validate_results=True
        )
        
        assert config.k == 10
        assert config.search_params == {'nprobe': 32}
        assert config.parallel_search is True
        assert config.timeout_seconds == 30.0
        assert config.validate_results is True
        
    def test_invalid_k(self):
        """Test SearchConfig with invalid k"""
        with pytest.raises(ValueError, match="k must be positive"):
            SearchConfig(k=0)
            
    def test_invalid_timeout(self):
        """Test SearchConfig with invalid timeout"""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            SearchConfig(k=10, timeout_seconds=-1.0)


class TestSearchResultAggregator:
    """Test SearchResultAggregator class"""
    
    @pytest.fixture
    def mock_gpu_manager(self):
        """Create a mock GPU manager"""
        manager = Mock(spec=GPUResourceManager)
        manager.get_available_gpu_ids.return_value = [0, 1]
        manager.validate_gpu_index.return_value = True
        manager.get_safe_device_string.side_effect = lambda gpu_id: f'cuda:{gpu_id}'
        return manager
        
    @pytest.fixture
    def aggregator(self, mock_gpu_manager):
        """Create SearchResultAggregator instance"""
        return SearchResultAggregator(mock_gpu_manager)
        
    def test_initialization(self, mock_gpu_manager):
        """Test SearchResultAggregator initialization"""
        aggregator = SearchResultAggregator(mock_gpu_manager)
        
        assert aggregator.gpu_manager == mock_gpu_manager
        assert aggregator.search_history == []
        assert aggregator._active_searches == {}
        
    def test_validate_search_results_valid(self, aggregator):
        """Test validation of valid search results"""
        results = [
            SearchResult(
                distances=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                indices=np.array([[10, 20], [30, 40]], dtype=np.int64),
                gpu_id=0,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            ),
            SearchResult(
                distances=np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
                indices=np.array([[5, 15], [25, 35]], dtype=np.int64),
                gpu_id=1,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            )
        ]
        
        assert aggregator.validate_search_results(results, expected_queries=2, expected_k=2) is True
        
    def test_validate_search_results_empty(self, aggregator):
        """Test validation of empty search results"""
        with pytest.raises(ValueError, match="gpu_results cannot be empty"):
            aggregator.validate_search_results([], expected_queries=2, expected_k=2)
            
    def test_validate_search_results_nan_distances(self, aggregator):
        """Test validation with NaN distances"""
        results = [
            SearchResult(
                distances=np.array([[np.nan, 2.0]], dtype=np.float32),
                indices=np.array([[10, 20]], dtype=np.int64),
                gpu_id=0,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            )
        ]
        
        with pytest.raises(ValueError, match="contains NaN distances"):
            aggregator.validate_search_results(results, expected_queries=1, expected_k=2)
            
    def test_merge_search_results_single_gpu(self, aggregator):
        """Test merging results from a single GPU"""
        results = [
            SearchResult(
                distances=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                indices=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64),
                gpu_id=0,
                query_time=0.1,
                k_requested=3,
                k_returned=3
            )
        ]
        
        final_distances, final_indices = aggregator.merge_search_results(results, k=2)
        
        # Should return top-2 results for each query
        expected_distances = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32)
        expected_indices = np.array([[10, 20], [40, 50]], dtype=np.int64)
        
        np.testing.assert_array_equal(final_distances, expected_distances)
        np.testing.assert_array_equal(final_indices, expected_indices)
        
    def test_merge_search_results_multiple_gpus(self, aggregator):
        """Test merging results from multiple GPUs"""
        results = [
            SearchResult(
                distances=np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
                indices=np.array([[20, 40], [60, 80]], dtype=np.int64),
                gpu_id=0,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            ),
            SearchResult(
                distances=np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
                indices=np.array([[10, 30], [50, 70]], dtype=np.int64),
                gpu_id=1,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            )
        ]
        
        final_distances, final_indices = aggregator.merge_search_results(results, k=3)
        
        # Should merge and sort results globally
        expected_distances = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype=np.float32)
        expected_indices = np.array([[10, 20, 30], [50, 60, 70]], dtype=np.int64)
        
        np.testing.assert_array_equal(final_distances, expected_distances)
        np.testing.assert_array_equal(final_indices, expected_indices)
        
    def test_merge_search_results_empty_list(self, aggregator):
        """Test merging empty results list"""
        with pytest.raises(ValueError, match="Cannot merge empty results list"):
            aggregator.merge_search_results([], k=5)
            
    def test_merge_search_results_inconsistent_queries(self, aggregator):
        """Test merging results with inconsistent query counts"""
        results = [
            SearchResult(
                distances=np.array([[1.0, 2.0]], dtype=np.float32),  # 1 query
                indices=np.array([[10, 20]], dtype=np.int64),
                gpu_id=0,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            ),
            SearchResult(
                distances=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # 2 queries
                indices=np.array([[10, 20], [30, 40]], dtype=np.int64),
                gpu_id=1,
                query_time=0.1,
                k_requested=2,
                k_returned=2
            )
        ]
        
        with pytest.raises(ValueError, match="has.*queries, expected"):
            aggregator.merge_search_results(results, k=2)
            
    @patch('search_result_aggregator.CUVS_AVAILABLE', False)
    def test_simulate_search(self, aggregator):
        """Test simulated search operation"""
        query = torch.randn(2, 128)  # 2 queries, 128 dimensions
        
        distances, indices = aggregator._simulate_search(query, k=5)
        
        assert distances.shape == (2, 5)
        assert indices.shape == (2, 5)
        assert torch.all(distances >= 0)  # Distances should be non-negative
        assert torch.all(indices >= 0)   # Indices should be non-negative
        
        # Check that results are sorted by distance
        for i in range(2):
            assert torch.all(distances[i, :-1] <= distances[i, 1:])
            
    def test_perform_distributed_search_invalid_inputs(self, aggregator):
        """Test distributed search with invalid inputs"""
        config = SearchConfig(k=5)
        
        # Invalid query tensor
        with pytest.raises(ValueError, match="query must be a torch.Tensor"):
            aggregator.perform_distributed_search("not a tensor", {0: Mock()}, config)
            
        # Invalid query dimensions
        query_1d = torch.randn(128)  # 1D tensor
        with pytest.raises(ValueError, match="query must be 2D tensor"):
            aggregator.perform_distributed_search(query_1d, {0: Mock()}, config)
            
        # Empty query
        query_empty = torch.empty(0, 128)
        with pytest.raises(ValueError, match="query cannot be empty"):
            aggregator.perform_distributed_search(query_empty, {0: Mock()}, config)
            
        # Empty indices dict
        query = torch.randn(2, 128)
        with pytest.raises(ValueError, match="indices dictionary cannot be empty"):
            aggregator.perform_distributed_search(query, {}, config)
            
    def test_perform_distributed_search_invalid_gpu(self, aggregator, mock_gpu_manager):
        """Test distributed search with invalid GPU in indices"""
        mock_gpu_manager.validate_gpu_index.return_value = False
        
        query = torch.randn(2, 128)
        indices = {99: Mock()}  # Invalid GPU ID
        config = SearchConfig(k=5)
        
        with pytest.raises(ValueError, match="GPU 99 in indices is not available"):
            aggregator.perform_distributed_search(query, indices, config)
            
    @patch('search_result_aggregator.CUVS_AVAILABLE', False)
    def test_perform_distributed_search_sequential(self, aggregator, mock_gpu_manager):
        """Test sequential distributed search"""
        query = torch.randn(2, 128)
        indices = {0: Mock(), 1: Mock()}
        config = SearchConfig(k=3, parallel_search=False)
        
        # Mock GPU manager methods
        mock_gpu_manager.validate_gpu_index.return_value = True
        mock_gpu_manager.get_safe_device_string.side_effect = lambda gpu_id: f'cuda:{gpu_id}'
        
        result = aggregator.perform_distributed_search(query, indices, config)
        
        assert isinstance(result, AggregatedSearchResult)
        assert result.num_queries == 2
        assert result.k_requested == 3
        assert len(result.gpu_results) == 2
        assert result.final_distances.shape == (2, 3)
        assert result.final_indices.shape == (2, 3)
        
    def test_search_history_management(self, aggregator):
        """Test search history management"""
        # Initially empty
        assert aggregator.get_search_history() == []
        
        # Add mock result to history
        mock_result = Mock(spec=AggregatedSearchResult)
        aggregator.search_history.append(mock_result)
        
        history = aggregator.get_search_history()
        assert len(history) == 1
        assert history[0] == mock_result
        
        # Clear history
        aggregator.clear_search_history()
        assert aggregator.get_search_history() == []
        
    def test_active_searches_tracking(self, aggregator):
        """Test active searches tracking"""
        # Initially empty
        assert aggregator.get_active_searches() == {}
        
        # Add active search
        aggregator._active_searches[0] = True
        active = aggregator.get_active_searches()
        assert active == {0: True}
        
        # Modify returned dict shouldn't affect internal state
        active[1] = True
        assert aggregator.get_active_searches() == {0: True}
        
    def test_string_representations(self, aggregator, mock_gpu_manager):
        """Test string representations"""
        str_repr = str(aggregator)
        assert "SearchResultAggregator" in str_repr
        assert "history_size=0" in str_repr
        
        repr_str = repr(aggregator)
        assert "SearchResultAggregator" in repr_str
        assert "history_size=0" in repr_str
        assert "active_searches=0" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])