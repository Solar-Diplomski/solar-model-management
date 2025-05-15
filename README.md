# solar-model-management

## About solar-model-management service
solar-model-management service is responsible for managing machine learning models and their versions, specifically for predicting outputs of solar panel powerplants. It receives models submitted by users via the UI, saves them to the filesystem in an organized manner, and stores model metadata in a PostgreSQL database. This service enables version control, traceability, and efficient management of models used in the solar power prediction workflow.

Technical details:
Language: Python 3
Framework: FastAPI (for building RESTful APIs)
ASGI Server: Uvicorn (for running FastAPI applications)
Model Uploads: Handled via multipart/form-data requests
Filesystem Organization: Models are stored in a structured directory (e.g., by user, model name, and version)
Database: PostgreSQL (for storing model metadata such as name, version, upload date, user, and file path)
Core Libraries: Pydantic (data validation), python-multipart (file uploads)
Testing: Pytest
Development Tools: Black (code formatting), Flake8 (linting)

## Relationships with other services

### API Gateway
solar-model-management service receives requests exclusively from API Gateway. API Gateway is used to connect all services in the project ecosystem to UI. It is implemented in Java with reactive Spring Boot. It uses Spring Cloud Gateway. Technical details:
Language: Java 21
Framework: Spring Boot 3.4.4 (using the reactive WebFlux module)
Cloud Framework: Spring Cloud 2024.0.0
Core Functionality: Spring Cloud Gateway (acting as an API Gateway)
Security: Spring Security with OAuth2 Resource Server support (integrating with Auth0)
Build Tool: Apache Maven

### UI
Relevant UI function to this service is adding models trough solar-model-management service. Technical details of the project:
Project Type: Frontend Web Application
Core Technologies:
Language: TypeScript
Framework: React (react, react-dom)
Build Tool: Vite (vite, @vitejs/plugin-react)
Package Manager: npm
Key Frameworks & Libraries:
UI Development Framework: Refine (@refinedev/core) - This framework provides structure for building CRUD applications, handling data fetching, routing, and state management.
UI Component Library: Ant Design (antd, @refinedev/antd) - Used for pre-built UI components, integrated with Refine.
Routing: React Router v6 (react-router-dom, @refinedev/react-router-v6) - Managed via Refine's integration.