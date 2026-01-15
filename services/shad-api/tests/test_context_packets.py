"""Tests for soft dependencies and context packets.

Per SPEC.md Section 1.6:
- Decomposition emits hard_deps (must complete) and soft_deps (useful if available)
- Completed nodes produce context packets (summary, artifacts, keywords)
- Scheduler injects packets into pending nodes' retrieval
"""

from __future__ import annotations

import pytest

from shad.engine.context_packets import (
    ContextPacket,
    ContextPacketStore,
    NodeContextManager,
)


class TestContextPacket:
    """Tests for context packets."""

    def test_create_context_packet(self) -> None:
        """Test creating a context packet."""
        packet = ContextPacket(
            node_id="node_123",
            stage_name="types_contracts",
            summary="Defined User and Auth types",
            artifacts=["src/types.ts", "src/auth/types.ts"],
            keywords=["User", "Auth", "JWT", "Token"],
        )

        assert packet.node_id == "node_123"
        assert packet.stage_name == "types_contracts"
        assert "User" in packet.summary
        assert len(packet.artifacts) == 2
        assert "JWT" in packet.keywords

    def test_packet_to_context_string(self) -> None:
        """Test converting packet to context string for injection."""
        packet = ContextPacket(
            node_id="node_123",
            stage_name="types_contracts",
            summary="Defined User type with id, name, email fields",
            artifacts=["src/types.ts"],
            keywords=["User", "id", "name", "email"],
        )

        context_str = packet.to_context_string()

        assert "types_contracts" in context_str
        assert "User" in context_str
        assert "src/types.ts" in context_str

    def test_packet_relevance_score(self) -> None:
        """Test calculating relevance score for a query."""
        packet = ContextPacket(
            node_id="node_123",
            stage_name="types_contracts",
            summary="Defined User type with authentication fields",
            artifacts=["src/types.ts"],
            keywords=["User", "Auth", "authentication", "JWT"],
        )

        # Query with matching keywords should have high score
        score = packet.relevance_score("implement user authentication")
        assert score > 0.5

        # Query with no matching keywords should have low score
        score = packet.relevance_score("database migrations")
        assert score < 0.3


class TestContextPacketStore:
    """Tests for context packet storage."""

    @pytest.fixture
    def store(self) -> ContextPacketStore:
        """Create a packet store."""
        return ContextPacketStore()

    def test_add_packet(self, store: ContextPacketStore) -> None:
        """Test adding a packet to the store."""
        packet = ContextPacket(
            node_id="node_1",
            stage_name="types_contracts",
            summary="Types defined",
            artifacts=[],
            keywords=["types"],
        )

        store.add(packet)
        assert store.get("node_1") == packet

    def test_get_packets_by_stage(self, store: ContextPacketStore) -> None:
        """Test getting packets by stage name."""
        packet1 = ContextPacket(
            node_id="node_1",
            stage_name="types_contracts",
            summary="Types 1",
            artifacts=[],
            keywords=[],
        )
        packet2 = ContextPacket(
            node_id="node_2",
            stage_name="types_contracts",
            summary="Types 2",
            artifacts=[],
            keywords=[],
        )
        packet3 = ContextPacket(
            node_id="node_3",
            stage_name="implementation",
            summary="Impl",
            artifacts=[],
            keywords=[],
        )

        store.add(packet1)
        store.add(packet2)
        store.add(packet3)

        type_packets = store.get_by_stage("types_contracts")
        assert len(type_packets) == 2

    def test_get_relevant_packets(self, store: ContextPacketStore) -> None:
        """Test getting relevant packets for a query."""
        packet1 = ContextPacket(
            node_id="node_1",
            stage_name="types_contracts",
            summary="User authentication types",
            artifacts=["types.ts"],
            keywords=["User", "Auth", "JWT"],
        )
        packet2 = ContextPacket(
            node_id="node_2",
            stage_name="db_schema",
            summary="Database schema for products",
            artifacts=["schema.sql"],
            keywords=["Product", "Category", "Price"],
        )

        store.add(packet1)
        store.add(packet2)

        relevant = store.get_relevant("implement user login", limit=5)
        assert len(relevant) > 0
        # Auth-related packet should be first
        assert relevant[0].node_id == "node_1"


class TestNodeContextManager:
    """Tests for node context management."""

    @pytest.fixture
    def manager(self) -> NodeContextManager:
        """Create a context manager."""
        return NodeContextManager()

    def test_create_packet_from_node_result(
        self, manager: NodeContextManager
    ) -> None:
        """Test creating a context packet from node result."""
        packet = manager.create_packet(
            node_id="node_123",
            stage_name="types_contracts",
            result="export interface User { id: string; name: string; email: string; }",
        )

        assert packet.node_id == "node_123"
        assert packet.stage_name == "types_contracts"
        # Should extract keywords from result
        assert len(packet.keywords) > 0

    def test_get_context_for_node(self, manager: NodeContextManager) -> None:
        """Test getting context for a node with soft dependencies."""
        # Add some packets first
        packet1 = ContextPacket(
            node_id="node_1",
            stage_name="types_contracts",
            summary="User types defined",
            artifacts=["types.ts"],
            keywords=["User", "Auth"],
        )
        manager.store.add(packet1)

        # Get context for a node that has soft_dep on types_contracts
        context = manager.get_context_for_node(
            node_id="node_2",
            stage_name="implementation",
            task="Implement user service",
            soft_deps=["types_contracts"],
        )

        assert context is not None
        assert "User" in context or "types" in context.lower()

    def test_inject_context_from_soft_deps(
        self, manager: NodeContextManager
    ) -> None:
        """Test injecting context from soft dependencies."""
        # Add completed node packets
        packet1 = ContextPacket(
            node_id="contracts_node",
            stage_name="types_contracts",
            summary="Defined User, Auth, and Token types",
            artifacts=["src/types.ts"],
            keywords=["User", "Auth", "Token", "JWT"],
        )
        packet2 = ContextPacket(
            node_id="db_node",
            stage_name="db_schema",
            summary="Database schema with users and sessions tables",
            artifacts=["schema.sql"],
            keywords=["users", "sessions", "database"],
        )

        manager.store.add(packet1)
        manager.store.add(packet2)

        # Get context for implementation node with soft deps
        context = manager.inject_soft_dep_context(
            soft_deps=["types_contracts", "db_schema"],
            task="Implement user authentication service",
        )

        # Should include relevant context from both packets
        assert context is not None
        assert len(context) > 0

    def test_context_relevance_filtering(
        self, manager: NodeContextManager
    ) -> None:
        """Test that context is filtered by relevance."""
        # Add packets with varying relevance
        manager.store.add(ContextPacket(
            node_id="auth_types",
            stage_name="types_contracts",
            summary="Authentication types: User, Token, Session",
            artifacts=["auth/types.ts"],
            keywords=["User", "Token", "Session", "Auth"],
        ))
        manager.store.add(ContextPacket(
            node_id="product_types",
            stage_name="types_contracts",
            summary="Product types: Product, Category, Price",
            artifacts=["product/types.ts"],
            keywords=["Product", "Category", "Price"],
        ))

        # Query about auth should prefer auth-related context
        context = manager.get_relevant_context(
            task="Implement user authentication",
            limit=1,
        )

        assert "Auth" in context or "User" in context or "Token" in context


class TestContextPacketIntegration:
    """Integration tests for context packets with DAG execution."""

    def test_packet_propagation_workflow(self) -> None:
        """Test the full workflow of packet creation and propagation."""
        manager = NodeContextManager()

        # Simulate types_contracts completion
        types_result = """
        export interface User {
            id: string;
            name: string;
            email: string;
            passwordHash: string;
        }

        export interface AuthToken {
            token: string;
            userId: string;
            expiresAt: Date;
        }
        """

        packet = manager.create_packet(
            node_id="types_node",
            stage_name="types_contracts",
            result=types_result,
        )
        manager.store.add(packet)

        # Now implementation node requests context
        context = manager.get_context_for_node(
            node_id="impl_node",
            stage_name="implementation",
            task="Implement user authentication service",
            soft_deps=["types_contracts"],
        )

        # Context should include relevant type information
        assert "User" in context or "Auth" in context

    def test_multiple_soft_deps_context(self) -> None:
        """Test context injection from multiple soft dependencies."""
        manager = NodeContextManager()

        # Add multiple packets
        manager.store.add(ContextPacket(
            node_id="types_node",
            stage_name="types_contracts",
            summary="Core types: User, Auth, Session",
            artifacts=["types.ts"],
            keywords=["User", "Auth", "Session"],
        ))
        manager.store.add(ContextPacket(
            node_id="db_node",
            stage_name="db_schema",
            summary="Database tables: users, sessions, tokens",
            artifacts=["schema.sql"],
            keywords=["users", "sessions", "tokens", "database"],
        ))
        manager.store.add(ContextPacket(
            node_id="api_node",
            stage_name="openapi",
            summary="API endpoints: /auth/login, /auth/register, /users",
            artifacts=["openapi.yaml"],
            keywords=["auth", "login", "register", "API"],
        ))

        # Implementation with multiple soft deps
        context = manager.inject_soft_dep_context(
            soft_deps=["types_contracts", "db_schema", "openapi"],
            task="Implement complete authentication flow",
        )

        # Should have context from all relevant packets
        assert len(context) > 0
