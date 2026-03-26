"""Admin API routes for Skill CRUD."""
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session
from src.skills import repository as skill_repo

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------

class SkillStepSchema(BaseModel):
    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    loop_over: str | None = None


class SkillCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = None
    version: str = "1.0"
    triggers: dict[str, Any] | None = None
    steps: list[SkillStepSchema]
    is_active: bool = True


class SkillUpdateRequest(BaseModel):
    description: str | None = None
    version: str | None = None
    triggers: dict[str, Any] | None = None
    steps: list[SkillStepSchema] | None = None
    is_active: bool | None = None


class SkillResponse(BaseModel):
    id: str
    name: str
    version: str
    description: str | None
    triggers: dict[str, Any] | None
    steps: list[dict[str, Any]]
    is_active: bool

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _skill_to_response(skill) -> SkillResponse:
    return SkillResponse(
        id=str(skill.id),
        name=skill.name,
        version=skill.version,
        description=skill.description,
        triggers=skill.triggers,
        steps=skill.steps,
        is_active=skill.is_active,
    )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/skills", response_model=SkillResponse, status_code=201)
async def create_skill(
    body: SkillCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    """Create a new skill."""
    existing = await skill_repo.get_skill(session, name=body.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Skill '{body.name}' already exists")

    skill = await skill_repo.create_skill(
        session,
        name=body.name,
        steps=[s.model_dump() for s in body.steps],
        description=body.description,
        version=body.version,
        triggers=body.triggers,
        is_active=body.is_active,
    )
    await session.commit()
    return _skill_to_response(skill)


@router.get("/skills", response_model=list[SkillResponse])
async def list_skills(
    active_only: bool = True,
    session: AsyncSession = Depends(get_session),
):
    """List all skills."""
    skills = await skill_repo.list_skills(session, active_only=active_only)
    return [_skill_to_response(s) for s in skills]


@router.get("/skills/{name}", response_model=SkillResponse)
async def get_skill(
    name: str,
    session: AsyncSession = Depends(get_session),
):
    """Get a skill by name."""
    skill = await skill_repo.get_skill(session, name=name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    return _skill_to_response(skill)


@router.put("/skills/{name}", response_model=SkillResponse)
async def update_skill(
    name: str,
    body: SkillUpdateRequest,
    session: AsyncSession = Depends(get_session),
):
    """Update an existing skill."""
    existing = await skill_repo.get_skill(session, name=name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

    updates: dict[str, Any] = {}
    if body.description is not None:
        updates["description"] = body.description
    if body.version is not None:
        updates["version"] = body.version
    if body.triggers is not None:
        updates["triggers"] = body.triggers
    if body.steps is not None:
        updates["steps"] = [s.model_dump() for s in body.steps]
    if body.is_active is not None:
        updates["is_active"] = body.is_active

    skill = await skill_repo.update_skill(session, name=name, **updates)
    await session.commit()
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found after update")
    return _skill_to_response(skill)


@router.delete("/skills/{name}", status_code=200)
async def deactivate_skill(
    name: str,
    session: AsyncSession = Depends(get_session),
):
    """Deactivate a skill (soft delete)."""
    skill = await skill_repo.deactivate_skill(session, name=name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    await session.commit()
    return {"detail": f"Skill '{name}' deactivated", "name": name}
