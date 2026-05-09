// @ts-check
import { defineConfig } from 'astro/config';

const repo = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? '';
const isGitHubActions = process.env.GITHUB_ACTIONS === 'true';
const isUserSiteRepo = repo.endsWith('.github.io');
const customDomain = process.env.CUSTOM_DOMAIN;

const base = customDomain ? '/' : isGitHubActions && repo && !isUserSiteRepo ? `/${repo}` : '/';

const site = customDomain
	? `https://${customDomain}`
	: process.env.GITHUB_REPOSITORY_OWNER
		? isUserSiteRepo
			? `https://${process.env.GITHUB_REPOSITORY_OWNER}.github.io`
			: `https://${process.env.GITHUB_REPOSITORY_OWNER}.github.io/${repo}`
		: undefined;

export default defineConfig({
	site,
	base,
});
