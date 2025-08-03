"""
Unit tests for the EvolvAttention web interface components.
"""

import pytest
from pathlib import Path
import json
from unittest.mock import patch, MagicMock


class TestWebInterfaceComponents:
    """Test suite for web interface components."""
    
    @pytest.fixture
    def html_file(self):
        """Fixture for HTML file path."""
        return Path("resources/public/index.html")
    
    @pytest.fixture
    def css_file(self):
        """Fixture for CSS file path."""
        return Path("resources/public/index.css")
    
    @pytest.fixture
    def js_file(self):
        """Fixture for JavaScript file path."""
        return Path("resources/public/index.js")
    
    @pytest.fixture
    def server_file(self):
        """Fixture for server file path."""
        return Path("src/evolvattention/server.py")

    def test_html_file_exists(self, html_file):
        """Test that the HTML file exists."""
        assert html_file.exists(), "index.html should exist"
    
    def test_html_structure_sections(self, html_file):
        """Test that the HTML file has all required sections."""
        content = html_file.read_text()
        
        # Check for key sections
        assert "Barycenter Calculation" in content
        assert "Cosine Similarities" in content
        assert "Evolutionary Algorithm" in content
        assert "Attention Analysis" in content
        assert "Visualization" in content
    
    def test_html_form_elements(self, html_file):
        """Test that the HTML file has all required form elements."""
        content = html_file.read_text()
        
        # Check for form elements
        assert 'id="barycenter-strings"' in content
        assert 'id="cosine-strings"' in content
        assert 'id="target-strings"' in content
        assert 'id="attention-string"' in content
    
    def test_html_buttons(self, html_file):
        """Test that the HTML file has all required buttons."""
        content = html_file.read_text()
        
        # Check for buttons
        assert 'id="calculate-barycenter"' in content
        assert 'id="calculate-cosine"' in content
        assert 'id="initialize-evolution"' in content
        assert 'id="analyze-attention"' in content
    
    def test_css_file_exists(self, css_file):
        """Test that the CSS file exists."""
        assert css_file.exists(), "index.css should exist"
    
    def test_css_styles(self, css_file):
        """Test that the CSS file has required style classes."""
        content = css_file.read_text()
        
        # Check for key style classes
        assert ".container" in content
        assert ".card" in content
        assert ".btn" in content
        assert ".form-group" in content
        assert ".status" in content
        assert ".results" in content
    
    def test_css_responsive_design(self, css_file):
        """Test that the CSS includes responsive design features."""
        content = css_file.read_text()
        
        # Check for responsive design elements
        assert "@media" in content
        assert "grid-template-columns" in content
        assert "minmax" in content
    
    def test_js_file_exists(self, js_file):
        """Test that the JavaScript file exists."""
        assert js_file.exists(), "index.js should exist"
    
    def test_js_class_structure(self, js_file):
        """Test that the JavaScript has the required class structure."""
        content = js_file.read_text()
        
        # Check for key classes and methods
        assert "class EvolvAttentionUI" in content
        assert "constructor" in content
        assert "initializeEventListeners" in content
    
    def test_js_api_methods(self, js_file):
        """Test that the JavaScript has all required API methods."""
        content = js_file.read_text()
        
        # Check for API methods
        assert "calculateBarycenter" in content
        assert "calculateCosineSimilarities" in content
        assert "initializeEvolution" in content
        assert "analyzeAttention" in content
    
    def test_js_api_endpoints(self, js_file):
        """Test that the JavaScript references all required API endpoints."""
        content = js_file.read_text()
        
        # Check for API endpoints (using the actual patterns in the code)
        assert "/barycenter" in content
        assert "/cosine-similarities" in content
        assert "/evolution/initialize" in content
        assert "/evolution/step" in content
        assert "/attention/analyze" in content
        assert "/health" in content
    
    def test_js_error_handling(self, js_file):
        """Test that the JavaScript includes error handling."""
        content = js_file.read_text()
        
        # Check for error handling patterns
        assert "try" in content
        assert "catch" in content
        assert "finally" in content
    
    def test_server_file_exists(self, server_file):
        """Test that the server file exists."""
        assert server_file.exists(), "server.py should exist"
    
    def test_server_endpoints(self, server_file):
        """Test that the server has all required endpoints."""
        content = server_file.read_text()
        
        # Check for all required endpoints
        assert "@app.post(\"/api/v1/barycenter\")" in content
        assert "@app.post(\"/api/v1/cosine-similarities\")" in content
        assert "@app.post(\"/api/v1/evolution/initialize\")" in content
        assert "@app.post(\"/api/v1/evolution/step\")" in content
        assert "@app.get(\"/api/v1/evolution/status\")" in content
        assert "@app.post(\"/api/v1/attention/analyze\")" in content
        assert "@app.get(\"/health\")" in content
    
    def test_server_data_models(self, server_file):
        """Test that the server has required data models."""
        content = server_file.read_text()
        
        # Check for data models
        assert "class BarycenterRequest" in content
        assert "class CosineSimilaritiesRequest" in content
        assert "class EvolutionInitRequest" in content
        assert "class AttentionAnalyzeRequest" in content
    
    def test_server_state_management(self, server_file):
        """Test that the server has state management."""
        content = server_file.read_text()
        
        # Check for state management
        assert "class ServerState" in content
        assert "state = ServerState()" in content


class TestWebInterfaceFunctionality:
    """Test suite for web interface functionality."""
    
    @pytest.fixture
    def mock_fetch(self):
        """Mock fetch function for testing API calls."""
        with patch('builtins.fetch') as mock:
            yield mock
    
    def test_api_call_structure(self):
        """Test that API calls follow the expected structure."""
        # This would test the actual API call logic if we had access to the browser environment
        # For now, we test the structure in the JavaScript file
        js_file = Path("resources/public/index.js")
        content = js_file.read_text()
        
        # Check for proper API call structure
        assert "async apiCall" in content
        assert "fetch" in content
        assert "JSON.stringify" in content
        assert "response.json()" in content
    
    def test_error_handling_structure(self):
        """Test that error handling is properly implemented."""
        js_file = Path("resources/public/index.js")
        content = js_file.read_text()
        
        # Check for error handling patterns
        assert "throw new Error" in content
        assert "catch (error)" in content
        assert "showStatus" in content
    
    def test_loading_states(self):
        """Test that loading states are implemented."""
        js_file = Path("resources/public/index.js")
        content = js_file.read_text()
        
        # Check for loading state implementation
        assert "showLoading" in content
        assert "disabled = true" in content
        assert "disabled = false" in content


class TestWebInterfaceIntegration:
    """Test suite for web interface integration."""
    
    def test_file_dependencies(self):
        """Test that all required files exist and are properly linked."""
        html_file = Path("resources/public/index.html")
        css_file = Path("resources/public/index.css")
        js_file = Path("resources/public/index.js")
        
        # Check that all files exist
        assert html_file.exists()
        assert css_file.exists()
        assert js_file.exists()
        
        # Check that HTML references CSS and JS
        html_content = html_file.read_text()
        assert 'href="/index.css"' in html_content
        assert 'src="/index.js"' in html_content
    
    def test_server_static_files(self):
        """Test that server properly serves static files."""
        server_file = Path("src/evolvattention/server.py")
        content = server_file.read_text()
        
        # Check for static file serving routes
        assert "@app.get(\"/index.css\")" in content
        assert "@app.get(\"/index.js\")" in content
        assert "FileResponse(\"resources/public/index.css\")" in content
        assert "FileResponse(\"resources/public/index.js\")" in content
    
    def test_cors_configuration(self):
        """Test that CORS is properly configured."""
        server_file = Path("src/evolvattention/server.py")
        content = server_file.read_text()
        
        # Check for CORS configuration
        assert "CORSMiddleware" in content
        assert "allow_origins" in content


if __name__ == "__main__":
    pytest.main([__file__]) 