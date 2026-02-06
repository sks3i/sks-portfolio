import { withBase } from "./utils/helpers";                                                                                                                     

export type Image = {                                                               
    src: string;
    alt?: string;
    caption?: string;                                                           
};

export type Link = {
    text: string;
    href: string;
};

export type Hero = {
    eyebrowText?: string;
    title?: string;
    text?: string;
    image?: Image;
    actions?: Link[];
};

export type About = {
    title?: string;
    text?: string;
};

export type Blog = {
    description?: string;
};

export type ContactInfo = {
    title?: string;
    text?: string;
    email?: {
        text?: string;
        href?: string;
        email?: string;
    };
    socialProfiles?: {
        text?: string;
        href?: string;
    }[];
};

export type Subscribe = {
    title?: string;
    text?: string;
    formUrl: string;
};

export type SiteConfig = {
    website: string;
    logo?: Image;
    title: string;
    description: string;
    image?: Image;
    headerNavLinks?: Link[];
    footerNavLinks?: Link[];
    socialLinks?: Link[];
    hero?: Hero;
    about?: About;
    contactInfo?: ContactInfo;
    subscribe?: Subscribe;
    blog?: Blog;
    postsPerPage?: number;
    recentPostLimit: number;
    projectsPerPage?: number;
};

const siteConfig: SiteConfig = {
    website: 'https://sabarishks.com',
    title: 'Sabarish Kuduwa Sivanath',
    description: 'Senior ML Engineer | Architecting End-to-End AI Systems',
    image: {
        src: withBase('/assets/images/portfolio-preview.jpeg'),
        alt: 'Sabarish Kuduwa Sivanath - Senior ML Engineer Portfolio'
    },
    headerNavLinks: [
        {
            text: 'Home',
            href: withBase('/')
        },
        {
            text: 'Blog',
            href: withBase('/blog')
        },
        {
            text: 'About',
            href: withBase('/about')
        },
        {
            text: 'Contact',
            href: withBase('/contact')
        }
    ],
    footerNavLinks: [
        {
            text: 'About',
            href: withBase('/about')
        },
        {
            text: 'Contact',
            href: withBase('/contact')
        },
        {
            text: 'RSS Feed',
            href: withBase('/rss.xml')
        }
    ],
    socialLinks: [
        {
            text: 'LinkedIn',
            href: 'https://www.linkedin.com/in/sabarish-kuduwa-sivanath/'
        },
        {
            text: 'GitHub',
            href: 'https://github.com/sks3i'
        }
    ],
    hero: {
        eyebrowText: 'End-to-End AI Systems',
        title: 'Sabarish K. Sivanath',
        text: "Senior ML Engineer at Uber. I build foundation models that perceive the world with superhuman accuracy and robotic systems that interact with it at 20µm resolution.",
        image: {
            src: withBase('/assets/images/pixeltrue-space-discovery.svg'),
            alt: 'Technical architecture visualization'
        },
        actions: [
            {
                text: 'View Research',
                href: withBase('/blog')
            },
            {
                text: 'Contact Me',
                href: withBase('/contact')
            }
        ]
    },
    about: {
        title: 'The Bits & The Atoms',
        text: 'With over 10 years of experience in Machine Learning, Computer Vision, and NLP, I focus on the intersection of complex algorithms and physical systems. At Uber, I lead multi-modal document transcription efforts, achieving a 95% reduction in development time through foundation models. My background spans from SoC performance at Intel to 0.1mm-precision robotic perception at Ebots. Beyond the code, I am a builder of water-cooled PCs, a wildlife photographer, and a seeker of the perfect South Indian filter coffee decoction.',
    },
    contactInfo: {
        title: 'Let\'s Connect',
        text: "Whether you're interested in multi-modal models, robotic perception, or EB-1A collaboration, feel free to reach out.",
        email: {
            text: "Drop me an email for professional inquiries or technical discussions.",
            href: "mailto:saba_rish91@hotmail.com",
            email: "saba_rish91@hotmail.com"
        },
        socialProfiles: [
            {
                text: "LinkedIn",
                href: "https://www.linkedin.com/in/sabarish-kuduwa-sivanath/"
            },
            {
                text: "GitHub",
                href: "https://github.com/sks3i"
            },
            {
                text: "Instagram",
                href: "https://instagram.com/sks3i"
            }
        ]
    },
    subscribe: {
        title: 'Stay Updated',
        text: 'Occasional deep dives into JAX, model interpretability, and robotics.',
        formUrl: '#'
    },
    blog: {
        description: "Technical essays on JAX, interpretability, fundamentals, and the physics of perception."
    },
    postsPerPage: 5,
    recentPostLimit: 3
};

export default siteConfig;
