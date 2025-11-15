from typing import Any, cast
import json
import jsonschema
from json import JSONDecodeError
import re
from pathlib import Path
import logging


logger = logging.getLogger(__name__)

response_schema_path = Path(__file__).parent / "../schema/response.schema.json"
response_schema = json.load(open(response_schema_path))


class JSONExtractionError(ValueError): ...


def json_extractor_simple(
    response: str, keys: list[str]
) -> dict[str, (str | list | None)]:
    """
    Extracts JSON from the response string.

    Args:
        response (str): The response string.

    Raises:
        JSONExtractionError: If no valid JSON output can be found, this \
error is raised with a list of error messages from attempts at conversion \
to json

    Returns:
        dict[str, str|list[dict[str, Any]]]: The extracted JSON.
    """
    errors = []
    # Simple schema from keys
    schema = {
        "type": "object",
        "properties": {
            key: {"type": ["string", "array", "null"]} for key in keys
        },
        "required": keys,
        "additionalProperties": True,
    }
    try:
        jsonschema.validate((response_dict := json.loads(response)), schema)
        response_dict: dict[str, (str | list | None)]
        return response_dict
    except (jsonschema.ValidationError, JSONDecodeError):
        errors.append("Not valid JSON response, trying to extract")
    try:
        response_dict = json.loads(f"{{{response}}}")
        response_dict: dict[str, (str | list | None)]
        jsonschema.validate(response_dict, schema)
        return response_dict
    except (jsonschema.ValidationError, JSONDecodeError) as e:
        errors.append(f"Tried adding {{}} to output, still invalid: {e}")
    code_pattern = r"```json\n(.+)\n```"
    potential_responses = re.findall(
        code_pattern, response, re.DOTALL
    )  # in case there are multiple code blocks. This was a problem last year
    responses = []
    for response in potential_responses:
        try:
            response_dict = json.loads(response)
            responses.append(response_dict)
        except JSONDecodeError as e:
            errors.append(f"Not valid JSON: {e}")
    difference = 1
    poses = (response.find("["), response.find("{"))
    pos = min([p for p in poses if p != -1] or [-1])
    start = pos
    open = response[pos]

    closed = "}" if open == "{" else "]"
    errors.append("No output in format ```json\n...\n```")
    while start != -1:
        try:
            while difference > 0:
                # recursively find next matching bracket, open or closed, updating difference accordingly
                if pos + 1 >= len(response):
                    raise JSONExtractionError(
                        "No JSON output in remaining string"
                    )
                nexts = (
                    response.find(open, pos + 1),
                    response.find(closed, pos + 1),
                )
                next = min([p for p in nexts if p != -1] or [-1])
                if next == -1:
                    raise JSONExtractionError(
                        "No JSON output in remaining string"
                    )
                difference += 1 if response[next] == open else -1
                pos = next
            json.loads((section := response[start : pos + 1]))
            responses.append(section)
        except (JSONDecodeError, JSONExtractionError) as e:
            errors.append((f"{type(e)}{e} in {start}:{pos + 1}"))
        poses = (response.find("{", pos + 1), response.find("[", pos + 1))
        pos = min([p for p in poses if p != -1] or [-1])
        start = pos
    valid_responses = []
    if responses:
        for response in responses:
            response_dict = json.loads(response)
            # Should never have JSONDecodeError here, all responses have already been loaded
            try:
                jsonschema.validate(response_dict, schema)
                valid_responses.append(json.loads(response))
            except jsonschema.ValidationError as e:
                errors.append(
                    f"Schema validation error for one of the responses: {e}"
                )
    if not valid_responses:
        response = f"{{{response}}}"  # puts {} around the response to try and make it valid JSON
        try:
            response_dict = json.loads(response)
            try:
                jsonschema.validate(json.loads(response), schema)
                return response_dict
            except jsonschema.ValidationError:
                response_dict = cast(dict[str, Any], response_dict)
        except JSONDecodeError as e:
            errors.append(f"Not valid JSON even if {{}} are added: {e}")
        except JSONExtractionError as e:
            errors.append(f"Valid JSON, but not following schema: {e}")
        logger.debug(
            f"Errors in extraction, just outputting response thought and audio: {errors}"
        )

    return {}


def json_extractor_for_tool_caller(
    response: str,
    schema: dict = response_schema
) -> dict[str, str | list[dict[str, Any]]]:
    """
    Handle common errors by several LLMs when outputting json. This function

    NOTE:
    - These attempts would have to be modified if the schema changes
    - This handles the error caused by several built in structured output 
        strategies where the 1st JSON is chosen if multiple are found 
        (whereas LLMs sometimes a single JSON for each tool call
        and a final answer only at the end)

    Args:
        response (str): Raw LLM response

    Raises:
        JSONExtractionError: If no valid JSON output can be found, this
            error is raised with a list of error messages from attempts at conversion
            to json

    Returns:
        dict[str, str|list[dict[str, Any]]]: _description_
    """
    errors = []
    try:
        jsonschema.validate((response_dict := json.loads(response)), schema)
        return response_dict
    except (jsonschema.ValidationError, JSONDecodeError):
        errors.append("Not valid JSON response, trying to extract")
    try:
        response_dict = json.loads(f"{{{response}}}")
        jsonschema.validate(response_dict, schema)
        return response_dict
    except (jsonschema.ValidationError, JSONDecodeError) as e:
        errors.append(f"Tried adding {{}} to output, still invalid: {e}")
    code_pattern = r"```json\n(.+)\n```"
    potential_responses = re.findall(
        code_pattern, response, re.DOTALL
    )  # in case there are multiple code blocks. This was a problem last year
    responses = []
    for response in potential_responses:
        try:
            json.loads(response)
            responses.append(response)
        except JSONDecodeError as e:
            errors.append(f"Not valid JSON: {e}")
    difference = 1
    poses = (response.find("["), response.find("{"))
    pos = min([p for p in poses if p != -1] or [-1])
    start = pos
    open = response[pos]

    closed = "}" if open == "{" else "]"
    errors.append("No output in format ```json\n...\n```")
    while start != -1:
        try:
            while difference > 0:
                # recursively find next matching bracket, open or closed, updating difference accordingly
                if pos + 1 >= len(response):
                    raise JSONExtractionError(
                        "No JSON output in remaining string"
                    )
                nexts = (
                    response.find(open, pos + 1),
                    response.find(closed, pos + 1),
                )
                next = min([p for p in nexts if p != -1] or [-1])
                if next == -1:
                    raise JSONExtractionError(
                        "No JSON output in remaining string"
                    )
                difference += 1 if response[next] == open else -1
                pos = next
            json.loads((section := response[start : pos + 1]))
            responses.append(section)
        except (JSONDecodeError, JSONExtractionError) as e:
            errors.append((f"{type(e)}{e} in {start}:{pos + 1}"))
        poses = (response.find("{", pos + 1), response.find("[", pos + 1))
        pos = min([p for p in poses if p != -1] or [-1])
        start = pos
    valid_responses = []
    if responses:
        for response in responses:
            response_dict = json.loads(response)
            # Should never have JSONDecodeError here, all responses have already been loaded
            try:
                jsonschema.validate(response_dict, schema)
                valid_responses.append(json.loads(response))
            except jsonschema.ValidationError as e:
                errors.append(
                    f"Schema validation error for one of the responses: {e}"
                )
                if isinstance(response_dict, list):
                    response_dict = {
                        "tool_calls": response_dict,
                        "thought": "",
                        "audio": "",
                    }
                    return response_dict
                response_dict = cast(dict[str, Any], response_dict)
                if "tool_calls" in response_dict:
                    response_dict["thought"] = response_dict.get("thought", "")
                    response_dict["audio"] = response_dict.get("audio", "")
                    return response_dict
                else:
                    errors.append(
                        f"Not following our schema even if looking only for tool_calls: {e}"
                    )

    if not valid_responses:
        response = f"{{{response}}}"  # puts {} around the response to try and make it valid JSON
        try:
            response_dict = json.loads(response)
            try:
                jsonschema.validate(json.loads(response), schema)
            except jsonschema.ValidationError as e:
                response_dict = cast(dict[str, Any], response_dict)
                errors.append(
                    "Not following correct schema if {} are added %s" % e
                )
                if "tool_calls" in response_dict:
                    response_dict["thought"] = response_dict.get("thought", "")
                    response_dict["audio"] = response_dict.get("audio", "")
                    return response_dict
                else:
                    errors.append(
                        "Not following our schema despite every attempt tool_calls"
                    )
                    raise JSONExtractionError(
                        f"No JSON output following our schema. Errors: {errors}"
                    )
        except JSONDecodeError as e:
            errors.append(f"Not valid JSON even if {{}} are added: {e}")
        except JSONExtractionError as e:
            errors.append(f"Valid JSON, but not following schema: {e}")
        logger.debug(
            f"Errors in extraction, just outputting response thought and audio: {errors}"
        )
        valid_responses = [
            {
                "thought": response[1:-1],
                "tool_calls": [],
                "audio": response[1:-1],
            }
        ]
    return valid_responses[-1]
