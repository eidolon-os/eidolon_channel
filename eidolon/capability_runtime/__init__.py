"""Capability Runtime: executing an Owner's commands through the Providers that implement them.

Lives in the Channel Provider process as its own module, apart from any
transport. It holds runtime state only; Owner master data stays with System
Data, and the vocabulary and wire shapes stay in the SDK.
"""
