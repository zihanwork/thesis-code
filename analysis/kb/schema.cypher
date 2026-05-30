// Neo4j schema for the VirtualHome persistent knowledge graph.
// Run once after starting Neo4j: see scripts/build_knowledge_base.sh.

// ---- uniqueness constraints
CREATE CONSTRAINT scene_file_id IF NOT EXISTS
FOR (s:Scene) REQUIRE s.file_id IS UNIQUE;

CREATE CONSTRAINT object_unique IF NOT EXISTS
FOR (o:Object) REQUIRE (o.file_id, o.node_id) IS UNIQUE;

CREATE CONSTRAINT action_name IF NOT EXISTS
FOR (a:Action) REQUIRE a.name IS UNIQUE;

CREATE CONSTRAINT property_name IF NOT EXISTS
FOR (p:Property) REQUIRE p.name IS UNIQUE;

CREATE CONSTRAINT precondition_kind IF NOT EXISTS
FOR (p:Precondition) REQUIRE p.kind IS UNIQUE;

CREATE CONSTRAINT effect_kind IF NOT EXISTS
FOR (e:Effect) REQUIRE e.kind IS UNIQUE;

CREATE CONSTRAINT failure_id IF NOT EXISTS
FOR (f:FailureCase) REQUIRE f.uid IS UNIQUE;

CREATE CONSTRAINT task_template_name IF NOT EXISTS
FOR (t:TaskTemplate) REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT derived_rule_id IF NOT EXISTS
FOR (r:DerivedRule) REQUIRE r.uid IS UNIQUE;

// ---- helpful lookup indexes
CREATE INDEX object_class IF NOT EXISTS FOR (o:Object) ON (o.class_name);
CREATE INDEX failure_type IF NOT EXISTS FOR (f:FailureCase) ON (f.failure_type);
CREATE INDEX task_template_category IF NOT EXISTS FOR (t:TaskTemplate) ON (t.category);
CREATE INDEX derived_rule_source IF NOT EXISTS FOR (r:DerivedRule) ON (r.source);
